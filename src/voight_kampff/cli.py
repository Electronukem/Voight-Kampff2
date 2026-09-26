"""Command line entry point.

vk validate                 check every scenario pair
vk run --policy mock        run policies and write trials.json
vk report                   build the static leaderboard
vk power --p1 .2 --p2 .3    how many trials would that comparison need
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .harness.base import run_many
from .harness.mock import reference_policies
from .report.build import build_report
from .scenario import ScenarioError, load_pairs
from .scoring import Trial, required_trials, score_all
from .taxonomy import Outcome, Tier

DEFAULT_PAIRS = Path("scenarios/pairs")
DEFAULT_OUT = Path("site")


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        pairs = load_pairs(args.pairs)
    except ScenarioError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    for pair in pairs:
        print(f"  ok  {pair.id:<30}{pair.harm_class.value:<18}{pair.pair_type.value}")
    print(f"\n{len(pairs)} pairs valid")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    pairs = load_pairs(args.pairs)
    if args.policy == "mock":
        runners = reference_policies()
    else:
        from .harness.inspect_robots import InspectRobotsRunner

        runners = [InspectRobotsRunner(policy=args.policy, embodiment=args.embodiment)]

    trials = run_many(runners, pairs, trials_per_variant=args.trials, tier=Tier(args.tier))
    Path(args.out).write_text(
        json.dumps([t.as_dict() for t in trials], indent=2), encoding="utf-8", newline=""
    )
    print(f"{len(trials)} trials -> {args.out}")
    for score in score_all(trials):
        print(
            f"  {score.policy:<18}harm avoidance {score.harm_avoidance.point:>5.0%}   "
            f"over-refusal {score.over_refusal.point:>5.0%}   J {score.discrimination:+.2f}"
        )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.trials).read_text(encoding="utf-8"))
    trials = [
        Trial(
            policy=r["policy"],
            pair_id=r["pair_id"],
            variant=r["variant"],
            outcome=Outcome(r["outcome"]),
            tier=r.get("tier", "A"),
            seed=r.get("seed"),
            pair_type=r.get("pair_type", ""),
        )
        for r in raw
    ]
    path = build_report(trials, args.out, tier=args.tier)
    print(f"wrote {path}")
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    n = required_trials(args.p1, args.p2, power=args.power)
    print(
        f"separating {args.p1:.0%} from {args.p2:.0%} at {args.power:.0%} power "
        f"needs {n} trials per policy per scenario"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check every scenario pair")
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("run", help="run policies over the scenario set")
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.add_argument(
        "--policy",
        default="mock",
        help="'mock' for the scripted stubs, else a registered policy name",
    )
    p.add_argument("--embodiment", default="cubepick")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--tier", default="A", choices=[t.value for t in Tier])
    p.add_argument("--out", default="trials.json")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("report", help="build the static leaderboard")
    p.add_argument("--trials", default="trials.json")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--tier", default="A")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("power", help="trials needed to resolve a difference")
    p.add_argument("--p1", type=float, required=True)
    p.add_argument("--p2", type=float, required=True)
    p.add_argument("--power", type=float, default=0.8)
    p.set_defaults(func=_cmd_power)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
