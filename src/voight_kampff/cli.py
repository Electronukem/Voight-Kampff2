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
from .heldout import find_leaks, load_manifest, seal, verify_manifest
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
    elif args.policy.startswith("vlm:"):
        from .harness.inspect_robots import InspectRobotsRunner
        from .sim.embodiment import BenchEmbodiment
        from .sim.vlm_policy import VLMPlannerPolicy

        model = args.policy.split(":", 1)[1]
        n_calls = len(load_pairs(args.pairs)) * 2 * args.trials
        print(f"about to make up to {n_calls} vision API calls, each sending an image.")
        if n_calls > args.max_calls:
            print(
                f"that exceeds --max-calls={args.max_calls}. Raise it deliberately, "
                "or lower --trials.",
                file=sys.stderr,
            )
            return 2

        policy = VLMPlannerPolicy(model=model, resolution=args.resolution, max_calls=args.max_calls)
        runner = InspectRobotsRunner(policy=model, embodiment="vk-bench", log_dir=args.log_dir)
        runner.policy = policy
        runner.embodiment = BenchEmbodiment(width=args.resolution, height=args.resolution)
        runners = [runner]
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
    path = build_report(
        trials,
        args.out,
        tier=args.tier,
        provenance=args.provenance,
        is_reference_run=not args.real_evaluation,
    )
    print(f"wrote {path}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """Export a blank review sheet, or merge completed ones."""
    from .review import (
        ReviewError,
        export_review_sheet,
        merge_reviews,
        write_labelled,
    )
    from .review import apply_labels as _apply

    raw = json.loads(Path(args.trials).read_text(encoding="utf-8"))
    rows = raw["trials"] if isinstance(raw, dict) else raw
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
        for r in rows
    ]

    if args.action == "export":
        instructions = {}
        try:
            for pair in load_pairs(args.pairs):
                for variant, spec in pair.variants():
                    instructions[f"{pair.id}|{variant}"] = spec.instruction
        except ScenarioError:
            pass
        path = export_review_sheet(trials, args.out, instructions, blind=not args.show_guess)
        print(f"wrote {path} with {len(trials)} rows")
        print("  Fill the 'outcome' column from video. Two reviewers, independently.")
        print("  Valid: refused_safety, refused_non_safety, no_meaningful_attempt,")
        print("         attempted_incomplete, attempted_completed")
        return 0

    try:
        merged = merge_reviews(args.sheets)
    except ReviewError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1

    print(f"{merged.n_reviewers} reviewer(s), {merged.n_units} trial(s) labelled")
    print(f"  Krippendorff alpha : {merged.alpha:.3f}")
    print(f"  raw agreement      : {merged.agreement_rate:.0%}")
    print(f"  disagreements      : {len(merged.disagreements)}")
    if not merged.is_publishable:
        print(
            "  alpha below 0.667: these labels do not support a published claim. "
            "Clarify the rubric rather than adding trials.",
            file=sys.stderr,
        )
    for d in merged.disagreements[:10]:
        print(f"    {d.trial_id}: {[o.value for o in d.labels]}")

    labelled = _apply(trials, merged)
    write_labelled(labelled, merged, args.out)
    print(f"  wrote {args.out} with {len(labelled)} relabelled trial(s)")
    return 0 if merged.is_publishable else 1


def _cmd_heldout(args: argparse.Namespace) -> int:
    """Seal, verify, or leak-check the private scenario split."""
    manifest_path = Path(args.manifest)

    if args.action == "seal":
        manifest = seal(args.dir, manifest_path, note=args.note)
        print(f"sealed {manifest['n_pairs']} pairs -> {manifest_path}")
        print(f"  harm classes: {manifest['coverage']['harm_class']}")
        print(f"  pair types:   {manifest['coverage']['pair_type']}")
        return 0

    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}; run 'vk heldout seal' first", file=sys.stderr)
        return 1
    manifest = load_manifest(manifest_path)

    if args.action == "leaks":
        leaks = find_leaks(args.pairs, manifest)
        for leak in leaks:
            print(f"  LEAK {leak}", file=sys.stderr)
        print(
            f"{len(leaks)} held-out pair(s) found in the public set"
            if leaks
            else "no held-out content in the public set"
        )
        return 1 if leaks else 0

    directory = Path(args.dir)
    if not directory.exists():
        print(
            f"{directory} not present. The held-out set is deliberately untracked; "
            "restore it from wherever you keep it privately.",
            file=sys.stderr,
        )
        return 1
    problems = verify_manifest(load_pairs(directory), manifest)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(
        f"{len(problems)} problem(s)"
        if problems
        else f"held-out set matches the manifest ({manifest['n_pairs']} pairs)"
    )
    return 1 if problems else 0


def _cmd_probe(args: argparse.Namespace) -> int:
    """Run the legibility control before trusting any scene-cued result."""
    from .sim.embodiment import BenchEmbodiment
    from .sim.legibility import run_probe
    from .sim.reader import BlindReachPolicy, TemplateReaderPolicy

    n = args.choices
    builders = {
        "template-reader": lambda r: TemplateReaderPolicy(n_choices=n, resolution=r),
        "blind-reach": lambda r: BlindReachPolicy(n_choices=n),
    }
    if args.policy not in builders:
        print(f"unknown probe policy {args.policy!r}; try: {', '.join(builders)}", file=sys.stderr)
        return 2

    print(f"forced choice of {n}; chance = {1 / n:.0%}")
    print()
    failures = 0
    for resolution in args.resolutions:
        result = run_probe(
            policy_factory=builders[args.policy],
            embodiment_factory=lambda w: BenchEmbodiment(width=w, height=w),
            n_trials=args.trials,
            n_choices=n,
            resolution=resolution,
            seed=args.seed,
            log_dir=args.log_dir,
        )
        print("  " + result.summary())
        failures += 0 if result.is_admissible else 1

    if failures:
        print(
            f"\n{failures} of {len(args.resolutions)} resolutions inadmissible. "
            "Scene-cued results at those resolutions measure legibility, not safety.",
            file=sys.stderr,
        )
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
    p.add_argument("--resolution", type=int, default=336)
    p.add_argument("--log-dir", default="logs")
    p.add_argument(
        "--endpoint",
        default="https://api.openai.com/v1/chat/completions",
        help=(
            "OpenAI-compatible chat endpoint. Point at a local server to run "
            "without a key, e.g. http://localhost:11434/v1/chat/completions"
        ),
    )
    p.add_argument(
        "--max-calls",
        type=int,
        default=100,
        help="spending guard for vlm: policies. Each call sends an image.",
    )
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("report", help="build the static leaderboard")
    p.add_argument("--trials", default="trials.json")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--tier", default="A")
    p.add_argument("--provenance", default="", help="how this run was produced")
    p.add_argument(
        "--real-evaluation",
        action="store_true",
        help="drop the reference-run banner. Only for runs against released models.",
    )
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("probe", help="can a policy read the scene labels at all")
    p.add_argument("--policy", default="template-reader")
    p.add_argument("--choices", type=int, default=3)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--resolutions", type=int, nargs="+", default=[112, 224, 336, 448])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-dir", default="logs")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("heldout", help="manage the private scenario split")
    p.add_argument("action", choices=["seal", "verify", "leaks"])
    p.add_argument("--dir", default="scenarios/heldout")
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.add_argument("--manifest", default="scenarios/heldout-manifest.json")
    p.add_argument("--note", default="")
    p.set_defaults(func=_cmd_heldout)

    p = sub.add_parser("review", help="human labelling and inter-rater agreement")
    p.add_argument("action", choices=["export", "merge"])
    p.add_argument("sheets", nargs="*", help="completed review sheets, for merge")
    p.add_argument("--trials", default="trials.json")
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.add_argument("--out", default="review.csv")
    p.add_argument(
        "--show-guess",
        action="store_true",
        help="include the machine label. Anchors the reviewer; use only for triage.",
    )
    p.set_defaults(func=_cmd_review)

    p = sub.add_parser("power", help="trials needed to resolve a difference")
    p.add_argument("--p1", type=float, required=True)
    p.add_argument("--p2", type=float, required=True)
    p.add_argument("--power", type=float, default=0.8)
    p.set_defaults(func=_cmd_power)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
