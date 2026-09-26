"""Tests for human review and inter-rater agreement.

The alpha implementation is checked against a case computed by hand. A
reliability statistic that is silently wrong is worse than none: it would
license publishing labels nobody actually agrees on.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from voight_kampff.review import (
    REVIEW_COLUMNS,
    Disagreement,
    MergedReview,
    ReviewError,
    apply_labels,
    export_review_sheet,
    krippendorff_alpha,
    load_review_sheet,
    merge_reviews,
    parse_outcome,
)
from voight_kampff.scoring import Trial
from voight_kampff.taxonomy import Outcome

A = Outcome.REFUSED_SAFETY
B = Outcome.ATTEMPTED_COMPLETED


def _trials(n: int = 4) -> list[Trial]:
    return [
        Trial("p", "pair-x", "unsafe" if i % 2 == 0 else "benign", B, pair_type="scene-cued")
        for i in range(n)
    ]


class TestAlpha:
    def test_matches_a_hand_computed_case(self):
        """3 units, 2 raters: [A,A] [B,B] [A,B].

        o_AA=2, o_BB=2, o_AB=1, o_BA=1; n_A=n_B=3, n=6.
        Do = 2; De = (9+9)/5 = 3.6; alpha = 1 - 2/3.6 = 0.4444.
        """
        alpha = krippendorff_alpha({"1": [A, A], "2": [B, B], "3": [A, B]})
        assert alpha == pytest.approx(0.4444, abs=0.0005)

    def test_perfect_agreement_is_one(self):
        assert krippendorff_alpha({"1": [A, A], "2": [B, B]}) == pytest.approx(1.0)

    def test_systematic_disagreement_is_negative(self):
        """Worse than chance means different rubrics, not carelessness."""
        assert krippendorff_alpha({"1": [A, B], "2": [B, A], "3": [A, B]}) < 0

    def test_single_category_is_undefined_not_perfect(self):
        """Nobody drew a distinction, so there was nothing to agree about."""
        assert math.isnan(krippendorff_alpha({"1": [A, A], "2": [A, A]}))

    def test_single_rater_units_are_excluded(self):
        both = {"1": [A, A], "2": [B, B], "3": [A]}
        without = {"1": [A, A], "2": [B, B]}
        assert krippendorff_alpha(both) == pytest.approx(krippendorff_alpha(without))

    def test_no_ratings_is_undefined(self):
        assert math.isnan(krippendorff_alpha({}))

    def test_supports_more_than_two_raters(self):
        assert krippendorff_alpha({"1": [A, A, B], "2": [B, B, B]}) == pytest.approx(0.375)


class TestOutcomeParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("refused_safety", A),
            ("REFUSED_SAFETY", A),
            ("  rs  ", A),
            ("refused (safety)", A),
            ("attempted_completed", B),
            ("complete", B),
        ],
    )
    def test_accepts_documented_spellings(self, text, expected):
        assert parse_outcome(text) is expected

    def test_unknown_outcome_lists_the_valid_ones(self):
        with pytest.raises(ReviewError, match="refused_safety"):
            parse_outcome("probably fine")


class TestSheetRoundTrip:
    def test_export_leaves_outcome_blank(self, tmp_path: Path):
        path = export_review_sheet(_trials(), tmp_path / "r.csv")
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert len(rows) == 4
        assert all(r["outcome"] == "" for r in rows)

    def test_export_is_blind_by_default(self, tmp_path: Path):
        """Showing the machine guess anchors the reviewer."""
        path = export_review_sheet(_trials(), tmp_path / "r.csv")
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert all(r["machine_guess"] == "" for r in rows)

    def test_guess_can_be_shown_deliberately(self, tmp_path: Path):
        path = export_review_sheet(_trials(), tmp_path / "r.csv", blind=False)
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert all(r["machine_guess"] for r in rows)

    def test_trial_ids_are_unique(self, tmp_path: Path):
        path = export_review_sheet(_trials(6), tmp_path / "r.csv")
        ids = [r["trial_id"] for r in csv.DictReader(path.open(newline="", encoding="utf-8"))]
        assert len(set(ids)) == len(ids)

    def test_blank_rows_are_skipped_on_load(self, tmp_path: Path):
        path = export_review_sheet(_trials(), tmp_path / "r.csv")
        assert load_review_sheet(path) == {}

    def test_missing_columns_are_reported(self, tmp_path: Path):
        bad = tmp_path / "bad.csv"
        bad.write_text("trial_id,outcome\nx,complete\n", encoding="utf-8")
        with pytest.raises(ReviewError, match="missing columns"):
            load_review_sheet(bad)

    def test_bad_outcome_names_the_row(self, tmp_path: Path):
        path = export_review_sheet(_trials(1), tmp_path / "r.csv")
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        rows[0]["outcome"] = "nonsense"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(ReviewError, match=r":2:"):
            load_review_sheet(path)


def _sheet(tmp_path: Path, name: str, outcomes: list[str]) -> Path:
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for i, outcome in enumerate(outcomes):
            writer.writerow(
                {
                    "trial_id": f"p|pair-x|unsafe|{i}",
                    "pair_id": "pair-x",
                    "variant": "unsafe",
                    "pair_type": "scene-cued",
                    "expected": "refuse",
                    "instruction": "",
                    "machine_guess": "",
                    "outcome": outcome,
                    "reviewer_note": "",
                }
            )
    return path


class TestMerge:
    def test_agreeing_reviewers_produce_consensus(self, tmp_path: Path):
        a = _sheet(tmp_path, "a.csv", ["refused_safety", "attempted_completed"])
        b = _sheet(tmp_path, "b.csv", ["refused_safety", "attempted_completed"])
        merged = merge_reviews([a, b])
        assert merged.alpha == pytest.approx(1.0)
        assert not merged.disagreements
        assert merged.is_publishable

    def test_disagreements_are_reported_not_hidden(self, tmp_path: Path):
        a = _sheet(tmp_path, "a.csv", ["refused_safety", "attempted_completed"])
        b = _sheet(tmp_path, "b.csv", ["attempted_incomplete", "attempted_completed"])
        assert len(merge_reviews([a, b]).disagreements) == 1

    def test_a_two_way_tie_is_left_unresolved(self, tmp_path: Path):
        """Two reviewers splitting is a rubric problem, not noise to average."""
        a = _sheet(tmp_path, "a.csv", ["refused_safety"])
        b = _sheet(tmp_path, "b.csv", ["attempted_incomplete"])
        merged = merge_reviews([a, b])
        assert merged.labels == {}
        assert len(merged.disagreements) == 1

    def test_a_majority_resolves(self, tmp_path: Path):
        a = _sheet(tmp_path, "a.csv", ["refused_safety"])
        b = _sheet(tmp_path, "b.csv", ["refused_safety"])
        c = _sheet(tmp_path, "c.csv", ["attempted_incomplete"])
        assert merge_reviews([a, b, c]).labels["p|pair-x|unsafe|0"] is A

    def test_low_agreement_is_not_publishable(self, tmp_path: Path):
        a = _sheet(tmp_path, "a.csv", ["refused_safety", "refused_safety", "attempted_completed"])
        b = _sheet(
            tmp_path, "b.csv", ["attempted_completed", "attempted_incomplete", "refused_safety"]
        )
        assert not merge_reviews([a, b]).is_publishable

    def test_no_sheets_is_an_error(self):
        with pytest.raises(ReviewError, match="no review sheets"):
            merge_reviews([])


class TestApplyLabels:
    def test_unreviewed_trials_are_dropped(self):
        """A run reported as human-reviewed must not carry machine labels."""
        merged = MergedReview(
            labels={"p|pair-x|unsafe|0": A}, alpha=1.0, n_units=1, n_reviewers=2, disagreements=[]
        )
        out = apply_labels(_trials(4), merged)
        assert len(out) == 1
        assert out[0].outcome is A

    def test_metadata_is_preserved(self):
        merged = MergedReview(
            labels={"p|pair-x|unsafe|0": A}, alpha=1.0, n_units=1, n_reviewers=2, disagreements=[]
        )
        assert apply_labels(_trials(2), merged)[0].pair_type == "scene-cued"


class TestAgreementRate:
    def test_reported_alongside_alpha(self):
        merged = MergedReview(
            labels={},
            alpha=0.7,
            n_units=10,
            n_reviewers=2,
            disagreements=[Disagreement("x", [A, B], [])],
        )
        assert merged.agreement_rate == pytest.approx(0.9)
