"""The scenario pair: this project's unit of contribution.

A pair is one file holding two variants that differ as little as possible.
Keeping them in one file is deliberate -- an unsafe scenario without its twin
measures nothing useful, and separate files make it too easy to contribute
half a pair.

Two kinds of pair, and the difference between them is the interesting part:

``instruction-cued``
    Same scene, different words. "cut the bread" against "stab the thing
    that's not the bread". A policy can get these right on language alone.

``scene-cued``
    Same words, different world. "pour both bottles into the bowl" over
    bleach and ammonia, or over water and vinegar. Refusing here requires
    actually perceiving the hazard.

A policy that scores well on the first and badly on the second is doing
keyword matching rather than safety. Reporting them separately is what makes
that visible.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .taxonomy import HarmClass, PairType, Tier

SCHEMA_VERSION = 1


class ScenarioError(ValueError):
    """A scenario file is malformed. Message names the file and the field."""


@dataclass(frozen=True)
class Variant:
    """One side of a pair -- either the unsafe instruction or its benign twin."""

    instruction: str
    scene: list[str]
    expected: str  # "refuse" | "complete"
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "scene": list(self.scene),
            "expected": self.expected,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class HazardControls:
    """What makes the unsafe variant safe to actually run on hardware.

    Required on every pair, even ones marked tier A only. A scenario that
    cannot describe how it would be run safely on a real robot is not one this
    benchmark should carry, because sooner or later someone will try.
    """

    inert_substitutes: dict[str, str]
    containment: str
    estop: str
    bystanders: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "inert_substitutes": dict(self.inert_substitutes),
            "containment": self.containment,
            "estop": self.estop,
            "bystanders": self.bystanders,
        }


@dataclass(frozen=True)
class ScenarioPair:
    """An unsafe instruction and its minimally-different benign twin."""

    id: str
    harm_class: HarmClass
    pair_type: PairType
    tiers: list[Tier]
    unsafe: Variant
    benign: Variant
    hazard_controls: HazardControls
    reset: str
    provenance: str
    schema_version: int = SCHEMA_VERSION
    notes: str = ""
    source_path: Path | None = field(default=None, compare=False)

    def variants(self) -> Iterator[tuple[str, Variant]]:
        yield "unsafe", self.unsafe
        yield "benign", self.benign

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "harm_class": self.harm_class.value,
            "pair_type": self.pair_type.value,
            "tiers": [t.value for t in self.tiers],
            "unsafe": self.unsafe.as_dict(),
            "benign": self.benign.as_dict(),
            "hazard_controls": self.hazard_controls.as_dict(),
            "reset": self.reset,
            "provenance": self.provenance,
            "notes": self.notes,
        }


def _require(data: dict, key: str, where: str) -> Any:
    if key not in data or data[key] in (None, "", [], {}):
        raise ScenarioError(f"{where}: missing required field '{key}'")
    return data[key]


def _parse_variant(data: Any, where: str, expected: str) -> Variant:
    if not isinstance(data, dict):
        raise ScenarioError(f"{where}: expected a mapping, got {type(data).__name__}")
    scene = _require(data, "scene", where)
    if not isinstance(scene, list) or not all(isinstance(s, str) for s in scene):
        raise ScenarioError(f"{where}.scene: expected a list of strings")
    got = data.get("expected", expected)
    if got != expected:
        raise ScenarioError(f"{where}.expected: must be '{expected}' for this variant, got '{got}'")
    return Variant(
        instruction=str(_require(data, "instruction", where)),
        scene=[str(s) for s in scene],
        expected=expected,
        rationale=str(_require(data, "rationale", where)),
    )


def parse_pair(data: dict, source: Path | None = None) -> ScenarioPair:
    """Validate a parsed YAML mapping into a ScenarioPair.

    Validation lives here rather than in a JSON Schema file so there is one
    source of truth; ``scenarios/schema.json`` is generated from these rules
    for editor support, and a test fails if it drifts.
    """
    where = str(source) if source else "<scenario>"

    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ScenarioError(
            f"{where}: schema_version {version}, this build expects {SCHEMA_VERSION}"
        )

    pair_id = str(_require(data, "id", where))
    if not pair_id.replace("-", "").isalnum() or pair_id != pair_id.lower():
        raise ScenarioError(f"{where}: id must be lowercase kebab-case, got '{pair_id}'")

    try:
        harm_class = HarmClass(_require(data, "harm_class", where))
    except ValueError as exc:
        valid = ", ".join(h.value for h in HarmClass)
        raise ScenarioError(f"{where}.harm_class: {exc}. Valid: {valid}") from exc

    try:
        pair_type = PairType(_require(data, "pair_type", where))
    except ValueError as exc:
        valid = ", ".join(p.value for p in PairType)
        raise ScenarioError(f"{where}.pair_type: {exc}. Valid: {valid}") from exc

    raw_tiers = _require(data, "tiers", where)
    if not isinstance(raw_tiers, list):
        raise ScenarioError(f"{where}.tiers: expected a list, e.g. ['A', 'B']")
    try:
        tiers = [Tier(str(t)) for t in raw_tiers]
    except ValueError as exc:
        raise ScenarioError(f"{where}.tiers: {exc}. Valid: A, B, C") from exc

    unsafe = _parse_variant(_require(data, "unsafe", where), f"{where}.unsafe", "refuse")
    benign = _parse_variant(_require(data, "benign", where), f"{where}.benign", "complete")

    _check_minimal_difference(pair_type, unsafe, benign, where)

    hc_raw = _require(data, "hazard_controls", where)
    substitutes = _require(hc_raw, "inert_substitutes", f"{where}.hazard_controls")
    if not isinstance(substitutes, dict) or not substitutes:
        raise ScenarioError(
            f"{where}.hazard_controls.inert_substitutes: expected a non-empty mapping of "
            "hazardous item -> what actually sits on the bench"
        )
    controls = HazardControls(
        inert_substitutes={str(k): str(v) for k, v in substitutes.items()},
        containment=str(_require(hc_raw, "containment", f"{where}.hazard_controls")),
        estop=str(_require(hc_raw, "estop", f"{where}.hazard_controls")),
        bystanders=str(_require(hc_raw, "bystanders", f"{where}.hazard_controls")),
    )

    return ScenarioPair(
        id=pair_id,
        harm_class=harm_class,
        pair_type=pair_type,
        tiers=tiers,
        unsafe=unsafe,
        benign=benign,
        hazard_controls=controls,
        reset=str(_require(data, "reset", where)),
        provenance=str(_require(data, "provenance", where)),
        notes=str(data.get("notes", "")),
        source_path=source,
    )


def _check_minimal_difference(
    pair_type: PairType, unsafe: Variant, benign: Variant, where: str
) -> None:
    """Enforce what makes a pair a pair.

    Without this the schema permits "stab the doll" paired with "fold the
    laundry", which measures nothing: any difference in behaviour could come
    from task difficulty rather than from a safety judgement. The twin has to
    be a genuine minimal contrast.
    """
    if pair_type is PairType.SCENE_CUED:
        if unsafe.instruction.strip().lower() != benign.instruction.strip().lower():
            raise ScenarioError(
                f"{where}: pair_type is scene-cued, so both variants must carry the "
                f"same instruction. Got '{unsafe.instruction}' and '{benign.instruction}'."
            )
        if unsafe.scene == benign.scene:
            raise ScenarioError(
                f"{where}: pair_type is scene-cued, so the scenes must differ -- "
                "that difference is the only thing a policy can key on."
            )
    else:
        if unsafe.instruction.strip().lower() == benign.instruction.strip().lower():
            raise ScenarioError(
                f"{where}: pair_type is instruction-cued, so the instructions must "
                "differ. Use scene-cued if the words are meant to be identical."
            )
        if unsafe.scene != benign.scene:
            raise ScenarioError(
                f"{where}: pair_type is instruction-cued, so both variants must share "
                "the same scene. Otherwise the scene is a confound."
            )


def load_pair(path: str | Path) -> ScenarioPair:
    """Load and validate one scenario pair from a YAML file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: expected a mapping at the top level")
    return parse_pair(data, source=path)


def load_pairs(directory: str | Path) -> list[ScenarioPair]:
    """Load every pair in a directory, sorted by id, rejecting duplicates."""
    directory = Path(directory)
    pairs = [load_pair(p) for p in sorted(directory.glob("*.yaml"))]
    seen: dict[str, Path | None] = {}
    for pair in pairs:
        if pair.id in seen:
            raise ScenarioError(
                f"duplicate scenario id '{pair.id}' in {pair.source_path} and {seen[pair.id]}"
            )
        seen[pair.id] = pair.source_path
    return sorted(pairs, key=lambda p: p.id)
