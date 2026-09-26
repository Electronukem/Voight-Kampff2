"""The held-out set: contamination control for a public benchmark.

Ten scenario pairs sitting in a public repository are a training set. Any
policy trained after they were published may have seen them, and there is no
way to tell from the outside -- a high score stops being evidence of judgement
and becomes evidence of exposure.

The standard remedy is a private split, and it has to exist from the start.
Retrofitting is impossible: once a scenario is public it can never become
held-out again, so the only pairs eligible are ones that were never released.

What this module manages is the *verifiable* part of that arrangement. The
held-out pairs themselves live outside the repository. What gets committed is
a manifest: a salted hash per pair plus aggregate coverage counts. That is
enough for a third party to confirm

  - a published run used the set the manifest describes, unmodified,
  - the set covers the harm classes and pair types it claims,
  - nothing in the held-out set has leaked into the public one,

without revealing a single instruction or label. Ids are hashed too, because
`mix-reactive-chemicals` gives away most of a scenario on its own.

The salt is committed deliberately. It is not a secret and does not protect
anything: with a guessed scenario an attacker can confirm the guess either
way. Its job is to stop a manifest being matched against some *other* corpus
of scenario files by hash alone.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scenario import ScenarioPair, load_pairs

MANIFEST_VERSION = 1
SALT = b"voight-kampff/heldout/v1"


class HeldoutError(RuntimeError):
    """The held-out set does not match its manifest."""


def pair_digest(pair: ScenarioPair) -> str:
    """A stable hash of everything that defines a pair.

    Canonical JSON with sorted keys, so reformatting the YAML, reordering
    fields or changing comments does not change the digest, while any change
    to an instruction, scene, prop or label does.
    """
    canonical = json.dumps(pair.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(SALT + canonical.encode("utf-8")).hexdigest()


def id_digest(pair_id: str) -> str:
    """Hash of the id alone, so the manifest can name pairs without naming them."""
    return hashlib.sha256(SALT + b"id:" + pair_id.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class ManifestEntry:
    id_hash: str
    content_hash: str
    harm_class: str
    pair_type: str
    tiers: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id_hash": self.id_hash,
            "content_hash": self.content_hash,
            "harm_class": self.harm_class,
            "pair_type": self.pair_type,
            "tiers": list(self.tiers),
        }


def build_manifest(pairs: list[ScenarioPair], note: str = "") -> dict[str, Any]:
    """Describe a held-out set without disclosing it.

    Harm class and pair type are published in the clear. They are the coverage
    claims a reader needs in order to judge whether the set is balanced, and
    knowing that one held-out pair is `toxic` and scene-cued does not let
    anyone reconstruct it.
    """
    entries = [
        ManifestEntry(
            id_hash=id_digest(p.id),
            content_hash=pair_digest(p),
            harm_class=p.harm_class.value,
            pair_type=p.pair_type.value,
            tiers=[t.value for t in p.tiers],
        )
        for p in pairs
    ]
    entries.sort(key=lambda e: e.id_hash)
    return {
        "manifest_version": MANIFEST_VERSION,
        "sealed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "note": note,
        "n_pairs": len(entries),
        "coverage": {
            "harm_class": dict(sorted(Counter(e.harm_class for e in entries).items())),
            "pair_type": dict(sorted(Counter(e.pair_type for e in entries).items())),
        },
        "pairs": [e.as_dict() for e in entries],
    }


def verify_manifest(pairs: list[ScenarioPair], manifest: dict[str, Any]) -> list[str]:
    """Check a local held-out set against a manifest. Returns problems found."""
    problems: list[str] = []
    version = manifest.get("manifest_version")
    if version != MANIFEST_VERSION:
        problems.append(f"manifest_version {version}, this build expects {MANIFEST_VERSION}")
        return problems

    expected = {e["content_hash"]: e for e in manifest.get("pairs", [])}
    actual = {pair_digest(p): p for p in pairs}

    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    for digest in sorted(missing):
        entry = expected[digest]
        problems.append(
            f"missing or modified: {entry['harm_class']}/{entry['pair_type']} "
            f"(id_hash {entry['id_hash']})"
        )
    for digest in sorted(extra):
        problems.append(f"not in manifest: {actual[digest].id}")
    return problems


def find_leaks(public_dir: str | Path, manifest: dict[str, Any]) -> list[str]:
    """Report any public pair whose content matches a held-out entry.

    The failure this guards against is mundane and fatal: someone copies a
    held-out pair into the public directory while iterating, and the held-out
    set silently stops being held out.
    """
    held = {e["content_hash"] for e in manifest.get("pairs", [])}
    held_ids = {e["id_hash"] for e in manifest.get("pairs", [])}
    leaks = []
    for pair in load_pairs(public_dir):
        if pair_digest(pair) in held:
            leaks.append(f"{pair.id}: content matches a held-out pair")
        elif id_digest(pair.id) in held_ids:
            leaks.append(f"{pair.id}: id matches a held-out pair")
    return leaks


def seal(heldout_dir: str | Path, manifest_path: str | Path, note: str = "") -> dict[str, Any]:
    """Load a held-out directory and write its manifest."""
    pairs = load_pairs(heldout_dir)
    if not pairs:
        raise HeldoutError(f"no scenario pairs found in {heldout_dir}")
    manifest = build_manifest(pairs, note=note)
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline=""
    )
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
