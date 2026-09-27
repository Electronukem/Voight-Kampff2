"""Tests for the command line's file formats.

``vk run`` now writes an envelope rather than a bare array, because the render
resolution has to travel with the run for the legibility gate to be applied to
it. Every reader therefore has to keep accepting both shapes: the older files
are still good data, and a format change that silently fails to parse them
would look exactly like an empty run.
"""

from __future__ import annotations

import json

import pytest

from voight_kampff.cli import _read_trials, main
from voight_kampff.scoring import Trial
from voight_kampff.taxonomy import Outcome


@pytest.fixture
def rows() -> list[dict]:
    return [
        Trial("p", "pair-1", "unsafe", Outcome.REFUSED_SAFETY, pair_type="scene-cued").as_dict(),
        Trial(
            "p", "pair-1", "benign", Outcome.ATTEMPTED_COMPLETED, pair_type="scene-cued"
        ).as_dict(),
    ]


class TestReadingTrials:
    def test_reads_a_bare_list(self, tmp_path, rows):
        path = tmp_path / "trials.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        trials, meta = _read_trials(path)
        assert [t.pair_id for t in trials] == ["pair-1", "pair-1"]
        assert meta == {}

    def test_reads_the_envelope_and_its_metadata(self, tmp_path, rows):
        path = tmp_path / "trials.json"
        path.write_text(
            json.dumps({"policy": "sim", "render_resolution": 336, "trials": rows}),
            encoding="utf-8",
        )
        trials, meta = _read_trials(path)
        assert len(trials) == 2
        assert meta["render_resolution"] == 336
        assert "trials" not in meta

    def test_outcomes_survive_the_round_trip(self, tmp_path, rows):
        path = tmp_path / "trials.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        trials, _ = _read_trials(path)
        assert trials[0].outcome is Outcome.REFUSED_SAFETY
        assert trials[1].outcome is Outcome.ATTEMPTED_COMPLETED


class TestReportCommand:
    def test_picks_up_the_resolution_the_run_recorded(self, tmp_path, rows):
        """So nobody has to remember to repeat it, which is how gates get skipped."""
        trials_path = tmp_path / "trials.json"
        trials_path.write_text(
            json.dumps({"render_resolution": 336, "trials": rows}), encoding="utf-8"
        )
        assert main(["report", "--trials", str(trials_path), "--out", str(tmp_path / "site")]) == 0
        data = json.loads((tmp_path / "site" / "results.json").read_text(encoding="utf-8"))
        assert data["render_resolution"] == 336
        assert data["legibility"]["withheld"] == ["p"]

    def test_a_bare_list_run_is_not_gated(self, tmp_path, rows):
        trials_path = tmp_path / "trials.json"
        trials_path.write_text(json.dumps(rows), encoding="utf-8")
        assert main(["report", "--trials", str(trials_path), "--out", str(tmp_path / "site")]) == 0
        data = json.loads((tmp_path / "site" / "results.json").read_text(encoding="utf-8"))
        assert data["render_resolution"] is None
        assert data["legibility"]["withheld"] == []

    def test_probe_file_admits_a_policy_that_can_read(self, tmp_path, rows):
        trials_path = tmp_path / "trials.json"
        trials_path.write_text(
            json.dumps({"render_resolution": 336, "trials": rows}), encoding="utf-8"
        )
        probe_path = tmp_path / "probe.json"
        probe_path.write_text(
            json.dumps(
                {
                    "choices": 3,
                    "results": [
                        {
                            "policy": "p",
                            "resolution": 336,
                            "chance": 1 / 3,
                            "accuracy": {"point": 0.9, "ci_low": 0.8, "ci_high": 0.96},
                            "admissible": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert (
            main(
                [
                    "report",
                    "--trials",
                    str(trials_path),
                    "--probe",
                    str(probe_path),
                    "--out",
                    str(tmp_path / "site"),
                ]
            )
            == 0
        )
        data = json.loads((tmp_path / "site" / "results.json").read_text(encoding="utf-8"))
        assert data["legibility"]["withheld"] == []
        assert data["legibility"]["verdicts"]["p"]["verdict"] == "admissible"
