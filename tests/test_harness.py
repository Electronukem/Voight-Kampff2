"""Tests for the trial loop and the scripted policies.

The stub policies double as executable specifications of what the metric is
supposed to distinguish. ``test_keyword_matcher_collapses_on_scene_cued_pairs``
is the one to watch: if it ever passes trivially, the scene-cued pairs have
stopped doing their job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voight_kampff.harness.base import run_many, run_suite
from voight_kampff.harness.mock import (
    AlwaysRefuse,
    Discriminating,
    KeywordMatcher,
    NeverRefuse,
    OverCautious,
    reference_policies,
)
from voight_kampff.scenario import load_pairs
from voight_kampff.scoring import score_all, score_by_pair_type, score_policy
from voight_kampff.taxonomy import Tier

PAIRS_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "pairs"


@pytest.fixture(scope="module")
def pairs():
    return load_pairs(PAIRS_DIR)


class TestRunSuite:
    def test_runs_both_variants_of_every_pair(self, pairs):
        trials = run_suite(NeverRefuse(), pairs, trials_per_variant=3)
        assert len(trials) == len(pairs) * 2 * 3
        assert {t.variant for t in trials} == {"unsafe", "benign"}

    def test_every_pair_is_represented_on_both_sides(self, pairs):
        trials = run_suite(NeverRefuse(), pairs, trials_per_variant=2)
        for pair in pairs:
            variants = {t.variant for t in trials if t.pair_id == pair.id}
            assert variants == {"unsafe", "benign"}, f"{pair.id} ran only {variants}"

    def test_seeds_are_unique(self, pairs):
        trials = run_suite(NeverRefuse(), pairs, trials_per_variant=5)
        seeds = [t.seed for t in trials]
        assert len(set(seeds)) == len(seeds)

    def test_results_are_reproducible(self, pairs):
        a = run_suite(Discriminating(), pairs, trials_per_variant=5)
        b = run_suite(Discriminating(), pairs, trials_per_variant=5)
        assert [t.outcome for t in a] == [t.outcome for t in b]

    def test_tier_filter_excludes_non_matching_pairs(self, pairs):
        assert run_suite(NeverRefuse(), pairs, 2, tier=Tier.HARDWARE)
        for pair in pairs:
            assert Tier.HARDWARE in pair.tiers

    def test_zero_trials_rejected(self, pairs):
        with pytest.raises(ValueError, match="at least 1"):
            run_suite(NeverRefuse(), pairs, trials_per_variant=0)

    def test_pair_type_is_recorded(self, pairs):
        trials = run_suite(NeverRefuse(), pairs, trials_per_variant=1)
        assert all(t.pair_type for t in trials)


class TestStubPolicyProperties:
    """What the leaderboard must say about each stub, or the metric is broken."""

    def test_always_refuse_scores_zero_discrimination(self, pairs):
        trials = run_suite(AlwaysRefuse(), pairs, trials_per_variant=10)
        score = score_policy("always-refuse", trials)
        assert score.harm_avoidance.point == 1.0
        assert score.discrimination == pytest.approx(0.0)

    def test_never_refuse_scores_zero_discrimination(self, pairs):
        trials = run_suite(NeverRefuse(), pairs, trials_per_variant=10)
        assert score_policy("never-refuse", trials).discrimination == pytest.approx(0.0)

    def test_discriminating_policy_ranks_first(self, pairs):
        trials = run_many(reference_policies(), pairs, trials_per_variant=20)
        assert score_all(trials)[0].policy == "discriminating"

    def test_over_cautious_loses_despite_perfect_harm_avoidance(self, pairs):
        """The single-axis winner must not be the overall winner."""
        trials = run_many([OverCautious(), Discriminating()], pairs, trials_per_variant=20)
        scores = {s.policy: s for s in score_all(trials)}
        assert scores["over-cautious"].harm_avoidance.point == 1.0
        assert (
            scores["over-cautious"].harm_avoidance.point
            > scores["discriminating"].harm_avoidance.point
        )
        assert scores["over-cautious"].discrimination < scores["discriminating"].discrimination

    def test_keyword_matcher_collapses_on_scene_cued_pairs(self, pairs):
        """The payoff of the whole scene-cued design.

        Scene-cued pairs carry identical instructions, so a policy reading only
        the words must answer both variants the same way. Its harm avoidance
        and over-refusal are then equal by construction and J is exactly zero,
        while on instruction-cued pairs it still looks respectable.
        """
        trials = run_suite(KeywordMatcher(), pairs, trials_per_variant=20)
        by_type = score_by_pair_type(trials)
        scene = by_type["scene-cued"][0]
        instruction = by_type["instruction-cued"][0]
        assert scene.discrimination == pytest.approx(0.0)
        assert instruction.discrimination > 0.2

    def test_capability_ceiling_produces_incomplete_trials(self, pairs):
        """A stub that never fails would hide the refusal/incapability distinction."""
        trials = run_suite(NeverRefuse(competence=0.5), pairs, trials_per_variant=20)
        outcomes = {t.outcome for t in trials}
        assert len(outcomes) > 1
