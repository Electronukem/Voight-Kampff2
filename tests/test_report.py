"""Tests for the static leaderboard.

Mostly guards against silent breakage in the generated HTML: an escaping bug or
a missing table would not fail any other test, and the table is the
accessibility path for readers the colour encoding does not reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voight_kampff.admissibility import Probe, ProbeSet
from voight_kampff.report.build import build_report, render_scatter, render_table
from voight_kampff.scoring import Trial, score_all
from voight_kampff.taxonomy import Outcome


@pytest.fixture
def trials() -> list[Trial]:
    out: list[Trial] = []
    for policy, unsafe_refusals, benign_refusals in [("good", 18, 1), ("cautious", 20, 20)]:
        for i in range(20):
            out.append(
                Trial(
                    policy,
                    "p1",
                    "unsafe",
                    Outcome.REFUSED_SAFETY if i < unsafe_refusals else Outcome.ATTEMPTED_COMPLETED,
                    pair_type="instruction-cued",
                )
            )
            out.append(
                Trial(
                    policy,
                    "p1",
                    "benign",
                    Outcome.REFUSED_SAFETY if i < benign_refusals else Outcome.ATTEMPTED_COMPLETED,
                    pair_type="instruction-cued",
                )
            )
    return out


class TestRendering:
    def test_scatter_is_well_formed_svg(self, trials):
        svg = render_scatter(score_all(trials))
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert svg.count("<svg") == svg.count("</svg>")

    def test_every_policy_gets_a_mark_and_a_label(self, trials):
        svg = render_scatter(score_all(trials))
        for policy in ("good", "cautious"):
            assert policy in svg

    def test_points_carry_hover_titles(self, trials):
        assert "<title>" in render_scatter(score_all(trials))

    def test_diagonal_reference_line_is_present(self, trials):
        assert "no discrimination" in render_scatter(score_all(trials))

    def test_table_includes_intervals(self, trials):
        assert "95% CI" in render_table(score_all(trials))

    def test_policy_names_are_escaped(self):
        """Scenario and policy names arrive from YAML and CLI flags."""
        trial = Trial("<script>x</script>", "p", "unsafe", Outcome.REFUSED_SAFETY)
        html = render_table(score_all([trial]))
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestBuildReport:
    def test_writes_both_files(self, trials, tmp_path: Path):
        path = build_report(trials, tmp_path)
        assert path.exists()
        assert (tmp_path / "results.json").exists()

    def test_results_json_round_trips(self, trials, tmp_path: Path):
        build_report(trials, tmp_path)
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert data["n_trials"] == len(trials)
        assert {s["policy"] for s in data["overall"]} == {"good", "cautious"}

    def test_trials_are_excluded_by_default(self, trials, tmp_path: Path):
        """Per-trial rows are ~30x the summary; they live in trials.json."""
        build_report(trials, tmp_path)
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert "trials" not in data

    def test_trials_can_be_included(self, trials, tmp_path: Path):
        build_report(trials, tmp_path, include_trials=True)
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert len(data["trials"]) == len(trials)

    def test_html_declares_both_themes(self, trials, tmp_path: Path):
        html = build_report(trials, tmp_path).read_text(encoding="utf-8")
        assert "prefers-color-scheme: dark" in html
        assert 'data-theme="dark"' in html

    def test_theme_variables_are_reachable_from_body(self, trials, tmp_path: Path):
        """Regression: vars declared on the wrapper leave the page chrome unthemed."""
        html = build_report(trials, tmp_path).read_text(encoding="utf-8")
        root_block = html.split("@media", 1)[0]
        assert "--page:" in root_block.split(".viz-root", 1)[0]


class TestProvenanceBanner:
    """A page headed 'leaderboard' reporting thousands of trials reads as results.

    Every number this project has produced comes from scripted policies. Being
    mistaken for evidence it does not have is the one failure a safety
    benchmark cannot recover from, so the banner is on by default and opting
    out has to be deliberate.
    """

    def test_banner_is_present_by_default(self, trials, tmp_path: Path):
        html = build_report(trials, tmp_path).read_text(encoding="utf-8")
        assert "not a model evaluation" in html.lower()
        assert "No released model has been evaluated" in html

    def test_title_does_not_claim_to_be_a_leaderboard(self, trials, tmp_path: Path):
        html = build_report(trials, tmp_path).read_text(encoding="utf-8")
        assert "<title>Voight-Kampff reference run</title>" in html

    def test_banner_names_the_policies(self, trials, tmp_path: Path):
        """So a reader can tell at a glance these are not model names."""
        html = build_report(trials, tmp_path).read_text(encoding="utf-8")
        assert "good" in html and "cautious" in html

    def test_provenance_text_is_shown_and_escaped(self, trials, tmp_path: Path):
        html = build_report(
            trials, tmp_path, provenance="ran <script>x</script> at 336px"
        ).read_text(encoding="utf-8")
        assert "at 336px" in html
        assert "<script>x</script>" not in html

    def test_results_json_records_the_flag(self, trials, tmp_path: Path):
        build_report(trials, tmp_path, provenance="how it ran")
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert data["is_reference_run"] is True
        assert data["provenance"] == "how it ran"

    def test_opting_out_is_explicit(self, trials, tmp_path: Path):
        path = build_report(trials, tmp_path, is_reference_run=False)
        html = path.read_text(encoding="utf-8")
        assert "not a model evaluation" not in html.lower()
        assert "<title>Voight-Kampff leaderboard</title>" in html
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert data["is_reference_run"] is False

    def test_committed_site_is_marked_as_a_reference_run(self):
        """Regression: the committed page once showed stub data unlabelled."""
        results = Path(__file__).resolve().parents[1] / "site" / "results.json"
        if not results.exists():
            pytest.skip("no committed site")
        data = json.loads(results.read_text(encoding="utf-8"))
        assert data.get("is_reference_run") is True, (
            "the committed site claims to be a real evaluation"
        )


def _probe(policy: str, resolution: int, ci_low: float) -> Probe:
    chance = 1 / 3
    return Probe(
        policy=policy,
        resolution=resolution,
        chance=chance,
        point=ci_low + 0.08,
        ci_low=ci_low,
        ci_high=min(1.0, ci_low + 0.15),
        admissible=ci_low > chance,
    )


@pytest.fixture
def mixed_trials() -> list[Trial]:
    """Two policies over both pair types: one that reads the bench, one blind."""
    out: list[Trial] = []
    for policy, unsafe_refusals in [("reader", 18), ("blind", 2)]:
        for pair_type in ("instruction-cued", "scene-cued"):
            for i in range(20):
                out.append(
                    Trial(
                        policy,
                        f"p-{pair_type}",
                        "unsafe",
                        Outcome.REFUSED_SAFETY
                        if i < unsafe_refusals
                        else Outcome.ATTEMPTED_COMPLETED,
                        pair_type=pair_type,
                    )
                )
                out.append(
                    Trial(
                        policy,
                        f"p-{pair_type}",
                        "benign",
                        Outcome.REFUSED_SAFETY if i < 1 else Outcome.ATTEMPTED_COMPLETED,
                        pair_type=pair_type,
                    )
                )
    return out


class TestLegibilityGate:
    """Scene-cued numbers are only publishable where a probe backs them.

    design.md ranks this as a live confound: the hazard on a scene-cued pair is
    legible only from rendered text, so a policy that cannot read the bench
    produces a low refusal rate for a reason that is not a safety failure. The
    two are indistinguishable in the outcome, which is exactly why the control
    has to be enforced here rather than left to whoever reads the page.
    """

    def test_unrendered_run_is_not_gated(self, mixed_trials, tmp_path: Path):
        """A scripted stub reads structured scene data and cannot misread a label."""
        html = build_report(mixed_trials, tmp_path).read_text(encoding="utf-8")
        assert "Legibility gate" not in html
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert data["legibility"]["gated_pair_types"] == []
        assert data["legibility"]["withheld"] == []

    def test_rendered_run_without_probes_withholds_everything(self, mixed_trials, tmp_path: Path):
        """No probe is not the same as a passing probe."""
        build_report(mixed_trials, tmp_path, resolution=336)
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert data["legibility"]["withheld"] == ["blind", "reader"]
        assert data["legibility"]["verdicts"]["reader"]["verdict"] == "unprobed"

    def test_passing_probe_publishes_and_failing_probe_withholds(
        self, mixed_trials, tmp_path: Path
    ):
        probes = ProbeSet((_probe("reader", 336, 0.80), _probe("blind", 336, 0.20)))
        html = build_report(mixed_trials, tmp_path, probes=probes, resolution=336).read_text(
            encoding="utf-8"
        )
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

        assert data["legibility"]["withheld"] == ["blind"]
        assert data["legibility"]["verdicts"]["reader"]["verdict"] == "admissible"
        assert data["legibility"]["verdicts"]["blind"]["verdict"] == "inadmissible"
        assert "withheld &mdash;" in html

    def test_withheld_policy_keeps_its_row(self, mixed_trials, tmp_path: Path):
        """Dropping the row would read as 'not run', which is a different claim."""
        probes = ProbeSet((_probe("blind", 336, 0.20),))
        html = build_report(mixed_trials, tmp_path, probes=probes, resolution=336).read_text(
            encoding="utf-8"
        )
        assert 'class="withheld"' in html
        assert "blind" in html

    def test_withheld_policy_is_absent_from_the_scene_cued_chart(
        self, mixed_trials, tmp_path: Path
    ):
        """A mark on a scatter asserts a position; there is no way to draw a caveat."""
        probes = ProbeSet((_probe("reader", 336, 0.80), _probe("blind", 336, 0.20)))
        html = build_report(mixed_trials, tmp_path, probes=probes, resolution=336).read_text(
            encoding="utf-8"
        )
        scene_svg = html.split('aria-label="scene-cued pairs"')[1].split("</svg>")[0]
        assert "reader" in scene_svg
        assert "blind" not in scene_svg

    def test_instruction_cued_results_are_untouched(self, mixed_trials, tmp_path: Path):
        """The hazard is in the words there, which a policy receives as text."""
        probes = ProbeSet((_probe("blind", 336, 0.20),))
        html = build_report(mixed_trials, tmp_path, probes=probes, resolution=336).read_text(
            encoding="utf-8"
        )
        instruction_svg = html.split('aria-label="instruction-cued pairs"')[1].split("</svg>")[0]
        assert "blind" in instruction_svg

    def test_gate_publishes_its_own_workings(self, mixed_trials, tmp_path: Path):
        """A control that silently removes rows is indistinguishable from a bug."""
        probes = ProbeSet((_probe("reader", 224, 0.80), _probe("blind", 336, 0.20)))
        html = build_report(mixed_trials, tmp_path, probes=probes, resolution=336).read_text(
            encoding="utf-8"
        )
        assert "Legibility gate" in html
        # Both policies listed, each with the measurement behind its verdict.
        assert "admissible" in html and "inadmissible" in html
        assert "224px" in html and "336px" in html

    def test_overall_numbers_are_flagged_as_still_pooling_withheld_trials(
        self, mixed_trials, tmp_path: Path
    ):
        probes = ProbeSet((_probe("blind", 336, 0.20),))
        html = build_report(mixed_trials, tmp_path, probes=probes, resolution=336).read_text(
            encoding="utf-8"
        )
        assert "pool both pair types" in html

    def test_render_resolution_is_recorded(self, mixed_trials, tmp_path: Path):
        build_report(mixed_trials, tmp_path, resolution=336)
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert data["render_resolution"] == 336
