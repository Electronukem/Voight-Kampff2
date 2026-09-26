"""The policy interface, and the loop that turns policies into trials.

Deliberately narrow: a runner sees one scenario variant and returns one
outcome. Everything about how the policy is hosted -- a VLA behind
XPolicyLab, an LLM planner, a simulator, a scripted stub -- lives behind this
line, so the scoring core never learns what a robot is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from ..scenario import ScenarioPair
from ..scoring import Trial
from ..taxonomy import Outcome, Tier


@runtime_checkable
class PolicyRunner(Protocol):
    """Anything that can be asked to act on a scenario variant."""

    @property
    def name(self) -> str:
        """Identifier as it appears on the leaderboard."""
        ...

    def run_trial(self, pair: ScenarioPair, variant: str, seed: int) -> Outcome:
        """Execute one trial and report how it ended."""
        ...


def run_suite(
    runner: PolicyRunner,
    pairs: Sequence[ScenarioPair],
    trials_per_variant: int = 20,
    tier: Tier = Tier.DECISION,
    start_seed: int = 0,
) -> list[Trial]:
    """Run every variant of every pair, ``trials_per_variant`` times each.

    Both variants are always run. Running only the unsafe side is the failure
    mode this whole design exists to prevent -- it produces a number that
    looks like safety and is satisfied by refusing everything.

    Seeds are deterministic and distinct per (pair, variant, repeat), so a run
    is reproducible and no two trials silently share an initial state.
    """
    if trials_per_variant < 1:
        raise ValueError("trials_per_variant must be at least 1")

    trials: list[Trial] = []
    seed = start_seed
    for pair in pairs:
        if tier not in pair.tiers:
            continue
        for variant, _ in pair.variants():
            for _ in range(trials_per_variant):
                outcome = runner.run_trial(pair, variant, seed)
                trials.append(
                    Trial(
                        policy=runner.name,
                        pair_id=pair.id,
                        variant=variant,
                        outcome=outcome,
                        tier=tier.value,
                        seed=seed,
                        pair_type=pair.pair_type.value,
                    )
                )
                seed += 1
    return trials


def run_many(
    runners: Iterable[PolicyRunner],
    pairs: Sequence[ScenarioPair],
    trials_per_variant: int = 20,
    tier: Tier = Tier.DECISION,
) -> list[Trial]:
    """Run a set of policies over the same scenarios, from the same seeds."""
    out: list[Trial] = []
    for runner in runners:
        out.extend(run_suite(runner, pairs, trials_per_variant, tier, start_seed=0))
    return out
