"""Tests for the static leaderboard.

Mostly guards against silent breakage in the generated HTML: an escaping bug or
a missing table would not fail any other test, and the table is the
accessibility path for readers the colour encoding does not reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
