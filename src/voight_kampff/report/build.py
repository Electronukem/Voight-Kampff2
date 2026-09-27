"""Static leaderboard: results JSON in, one self-contained HTML file out.

No JavaScript and no build step, because the leaderboard has to be as
reproducible as the numbers behind it -- anyone with the repo can regenerate it
and diff the output. Hover works through native SVG <title>, and the table
underneath is the real accessibility path, not a fallback.

The chart is a scatter rather than a ranked bar, and that is the whole
argument: a bar chart of "safety" has one axis, and the policy that tops it is
the one that refuses everything. Two axes force the trade-off into view. The
diagonal is where harm avoidance equals over-refusal -- every policy on that
line is exactly as likely to refuse a safe task as a dangerous one, which is to
say it is not making a safety decision at all.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..admissibility import GATED_PAIR_TYPES, ProbeSet, Ruling, rule_all
from ..scoring import PolicyScore, Trial, score_all, score_by_pair_type

# Validated against the light and dark chart surfaces with the palette checker
# (all-pairs mode, which is the gate scatter charts must clear). Slots 1 and 2.
SERIES = {
    "instruction-cued": {"light": "#2a78d6", "dark": "#3987e5"},
    "scene-cued": {"light": "#eb6834", "dark": "#d95926"},
    "overall": {"light": "#2a78d6", "dark": "#3987e5"},
}

_CSS = """
/* Roles are declared on :root, not on the chart wrapper. The page background
   is painted by <body>, which is an ancestor of the wrapper -- custom
   properties inherit downward only, so declaring them lower leaves the page
   chrome stuck on the browser default while the chart itself themes correctly. */
:root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7;
  --series-instruction: #2a78d6; --series-scene: #eb6834; --series-overall: #2a78d6;
  --status-serious: #ec835a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835;
    --series-instruction: #3987e5; --series-scene: #d95926; --series-overall: #3987e5;
  --status-serious: #ec835a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835;
  --series-instruction: #3987e5; --series-scene: #d95926; --series-overall: #3987e5;
  --status-serious: #ec835a;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.viz-root { max-width: 920px; margin: 0 auto; padding: 40px 24px 72px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 40px 0 6px; letter-spacing: -0.005em; }
.sub { color: var(--text-secondary); margin: 0 0 28px; }
.note { color: var(--text-secondary); font-size: 14px; margin: 6px 0 18px; }
figure {
  margin: 0 0 8px; background: var(--surface-1);
  border-radius: 10px; padding: 16px 8px 8px;
}
.facets { display: flex; flex-wrap: wrap; gap: 16px; }
.facets figure { flex: 1 1 380px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 8px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--text-secondary); font-weight: 600; font-size: 13px; }
td.ci { color: var(--text-secondary); font-variant-numeric: tabular-nums; }
td.num { font-variant-numeric: tabular-nums; }
.flag { color: var(--text-secondary); font-size: 13px; }
tr.withheld td { color: var(--text-muted); font-style: italic; }
tr.withheld td:first-child { font-style: normal; }
.banner {
  border: 2px solid var(--status-serious); border-radius: 10px;
  padding: 14px 18px; margin: 0 0 26px; background: var(--surface-1);
}
.banner .tag {
  display: inline-block; font-weight: 700; font-size: 12px; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--text-primary);
  border: 1.5px solid var(--status-serious); border-radius: 4px;
  padding: 2px 7px; margin-right: 10px;
}
.banner p { margin: 10px 0 0; color: var(--text-secondary); font-size: 14px; }
.banner strong { color: var(--text-primary); }
footer { margin-top: 44px; color: var(--text-muted); font-size: 13px; }
a { color: inherit; }
"""

W, H = 660, 400
PAD_L, PAD_R, PAD_T, PAD_B = 62, 26, 22, 52


def _x(value: float) -> float:
    return PAD_L + value * (W - PAD_L - PAD_R)


def _y(value: float) -> float:
    return H - PAD_B - value * (H - PAD_T - PAD_B)


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def _place_labels(points: Sequence[tuple[float, str]], min_gap: float = 15.0) -> list[float]:
    """Nudge overlapping direct labels apart, keeping their original order.

    Five policies routinely land within a few percent of each other on one
    axis, and unreadable labels would push the reader to the legend -- which
    for a single series does not exist.
    """
    order = sorted(range(len(points)), key=lambda i: points[i][0])
    placed = dict.fromkeys(range(len(points)), 0.0)
    last = float("-inf")
    for idx in order:
        wanted = points[idx][0]
        chosen = max(wanted, last + min_gap)
        placed[idx] = chosen
        last = chosen
    return [placed[i] for i in range(len(points))]


def render_scatter(scores: Sequence[PolicyScore], series: str = "overall", title: str = "") -> str:
    """Two-axis scatter with confidence whiskers on both axes."""
    colour = {
        "instruction-cued": "var(--series-instruction)",
        "scene-cued": "var(--series-scene)",
    }.get(series, "var(--series-overall)")

    parts: list[str] = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
        f'aria-label="{_esc(title or "Harm avoidance against over-refusal")}">'
    ]

    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gx, gy = _x(frac), _y(frac)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{_y(0):.1f}" x2="{gx:.1f}" y2="{_y(1):.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<line x1="{_x(0):.1f}" y1="{gy:.1f}" x2="{_x(1):.1f}" y2="{gy:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{_y(0) + 18:.1f}" text-anchor="middle" '
            f'font-size="12" fill="var(--text-muted)">{frac:.0%}</text>'
        )
        parts.append(
            f'<text x="{_x(0) - 10:.1f}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="var(--text-muted)">{frac:.0%}</text>'
        )

    # The line where harm avoidance equals over-refusal. Everything on it is
    # refusing safe and unsafe tasks at the same rate -- no safety decision at
    # all -- which is where a one-axis leaderboard would put its winner.
    parts.append(
        f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(1):.1f}" y2="{_y(1):.1f}" '
        f'stroke="var(--axis)" stroke-width="1.5" stroke-dasharray="5 4"/>'
        f'<text x="{_x(0.40):.1f}" y="{_y(0.40) + 19:.1f}" font-size="12" '
        f'fill="var(--text-muted)" transform="rotate(-33 {_x(0.40):.1f} {_y(0.40) + 19:.1f})">'
        "no discrimination (J = 0)</text>"
    )

    parts.append(
        f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(1):.1f}" y2="{_y(0):.1f}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
        f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(0):.1f}" y2="{_y(1):.1f}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
    )

    label_inputs = [(_y(s.harm_avoidance.point), s.policy) for s in scores]
    label_ys = _place_labels(label_inputs)

    for score, label_y in zip(scores, label_ys, strict=True):
        cx, cy = _x(score.over_refusal.point), _y(score.harm_avoidance.point)
        ha_lo, ha_hi = score.harm_avoidance.interval
        or_lo, or_hi = score.over_refusal.interval
        parts.append(
            f'<line x1="{cx:.1f}" y1="{_y(ha_lo):.1f}" x2="{cx:.1f}" y2="{_y(ha_hi):.1f}" '
            f'stroke="{colour}" stroke-width="2" opacity="0.45"/>'
            f'<line x1="{_x(or_lo):.1f}" y1="{cy:.1f}" x2="{_x(or_hi):.1f}" y2="{cy:.1f}" '
            f'stroke="{colour}" stroke-width="2" opacity="0.45"/>'
        )
        # 2px surface ring so overlapping marks stay separable.
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{colour}" '
            f'stroke="var(--surface-1)" stroke-width="2"/>'
        )
        tooltip = (
            f"{score.policy} - harm avoidance {score.harm_avoidance.point:.0%} "
            f"[{ha_lo:.0%}, {ha_hi:.0%}], over-refusal {score.over_refusal.point:.0%} "
            f"[{or_lo:.0%}, {or_hi:.0%}], J {score.discrimination:+.2f}"
        )
        # Generous invisible hit area: a 6px dot is not a hover target.
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="transparent">'
            f"<title>{_esc(tooltip)}</title></circle>"
        )
        anchor = "end" if cx > _x(0.72) else "start"
        dx = -12 if anchor == "end" else 12
        parts.append(
            f'<text x="{cx + dx:.1f}" y="{label_y + 4:.1f}" text-anchor="{anchor}" '
            f'font-size="13" fill="var(--text-primary)">{_esc(score.policy)}</text>'
        )

    parts.append(
        f'<text x="{_x(0.5):.1f}" y="{H - 10:.1f}" text-anchor="middle" font-size="13" '
        f'fill="var(--text-secondary)">over-refusal on benign twins -&gt;</text>'
        f'<text x="16" y="{_y(0.5):.1f}" text-anchor="middle" font-size="13" '
        f'fill="var(--text-secondary)" transform="rotate(-90 16 {_y(0.5):.1f})">'
        "harm avoidance on unsafe -&gt;</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def render_table(scores: Sequence[PolicyScore], rulings: Mapping[str, Ruling] | None = None) -> str:
    """The score table, withholding any row the legibility gate blocks.

    A withheld row keeps the policy name and says why instead of disappearing.
    Dropping it would read as "this policy was not run", which is a different
    and weaker claim than "this number cannot be interpreted".
    """
    rows = []
    for score in scores:
        ruling = (rulings or {}).get(score.policy)
        if ruling is not None and not ruling.verdict.is_publishable:
            rows.append(
                f'<tr class="withheld"><td>{_esc(score.policy)} '
                f'<span class="flag">({_esc(ruling.verdict.value)})</span></td>'
                f'<td colspan="7">withheld &mdash; {_esc(ruling.reason())}</td></tr>'
            )
            continue
        ha_lo, ha_hi = score.harm_avoidance.interval
        or_lo, or_hi = score.over_refusal.interval
        j_lo, j_hi = score.discrimination_interval
        flag = ' <span class="flag">(underpowered)</span>' if score.is_underpowered else ""
        rows.append(
            f"<tr><td>{_esc(score.policy)}{flag}</td>"
            f'<td class="num">{score.harm_avoidance.point:.0%}</td>'
            f'<td class="ci">[{ha_lo:.0%}, {ha_hi:.0%}]</td>'
            f'<td class="num">{score.over_refusal.point:.0%}</td>'
            f'<td class="ci">[{or_lo:.0%}, {or_hi:.0%}]</td>'
            f'<td class="num">{score.helpfulness.point:.0%}</td>'
            f'<td class="num">{score.discrimination:+.2f}</td>'
            f'<td class="ci">[{j_lo:+.2f}, {j_hi:+.2f}]</td></tr>'
        )
    return (
        "<table><thead><tr><th>policy</th><th>harm avoid.</th><th>95% CI</th>"
        "<th>over-refusal</th><th>95% CI</th><th>helpful</th><th>J</th><th>95% CI</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _gate_note(
    rulings: Mapping[str, Ruling], resolution: int | None, withheld: Sequence[str]
) -> str:
    """The gate's own workings, published alongside what it withheld.

    Every policy is listed, including the ones that passed, and each verdict
    carries the measurement behind it. A control that silently removes rows is
    indistinguishable from a bug, and a reader who cannot audit the gate has
    to take the withholding on trust.
    """
    if not rulings:
        return ""
    where = f"{resolution}px" if resolution else "an unrecorded render resolution"
    lead = (
        f"<strong>{len(withheld)} of {len(rulings)} policies</strong> had their "
        "scene-cued results withheld"
        if withheld
        else "<strong>Every policy cleared the gate</strong>"
    )
    rows = "".join(
        f"<tr><td>{_esc(policy)}</td><td>{_esc(ruling.verdict.value)}</td>"
        f"<td>{_esc(ruling.reason())}</td></tr>"
        for policy, ruling in sorted(rulings.items())
    )
    return f"""<h2>Legibility gate</h2>
<p class="note">Scene-cued pairs give both variants the identical instruction and
change only the world, so the hazard is legible only from what a label says. A
policy that cannot read the bench will fail to refuse for a reason that is not a
safety failure, and the two are indistinguishable in the outcome. So a scene-cued
number is published only where <code>vk probe</code> has shown that policy reading
labels above chance at {_esc(where)}; otherwise it is withheld as inadmissible
rather than reported as a negative result. {lead}.</p>
<table><thead><tr><th>policy</th><th>verdict</th><th>evidence</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def build_report(
    trials: Sequence[Trial],
    out_dir: str | Path,
    tier: str = "A",
    include_trials: bool = False,
    provenance: str = "",
    is_reference_run: bool = True,
    probes: ProbeSet | None = None,
    resolution: int | None = None,
) -> Path:
    """Write index.html and results.json. Returns the HTML path.

    Per-trial rows are left out by default: they run to hundreds of times the
    size of the summary, and the raw record already lives in trials.json.

    ``is_reference_run`` defaults to True and puts an unmissable banner at the
    top of the page. Every number this project has produced so far comes from
    scripted reference policies, not from any released model, and a page headed
    "leaderboard" reporting 2,000 trials reads as results whether or not that
    was intended. For a safety benchmark, being mistaken for evidence you do
    not have is the one error there is no recovering from -- so the default is
    to say so loudly, and a real evaluation has to opt out explicitly.

    ``probes`` and ``resolution`` are the legibility gate. Scene-cued pairs put
    the hazard entirely in rendered text, so a policy that cannot read the
    bench produces a low refusal rate for a reason that has nothing to do with
    safety. The gate turns on whenever ``resolution`` says the run rendered
    anything; from there, a missing probe withholds the scene-cued rows exactly
    as a failed one does, because no probe is not the same as a passing probe
    and only one of those two readings is safe to default to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = score_all(trials)
    by_type = score_by_pair_type(trials)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_pairs = len({t.pair_id for t in trials})

    # One ruling per policy, reused by every gated facet: the question the
    # probe answers is about the policy and the render resolution, not about
    # which slice of pairs is being drawn.
    rulings = rule_all(probes, [s.policy for s in overall], resolution)
    # The confound only exists where the policy saw the scene through pixels.
    # A render resolution is recorded exactly when something was rendered, so
    # its absence means the scripted-stub case -- an in-process policy handed
    # the scene as structured data, which cannot misread a label because it
    # never reads one. Gating those would withhold numbers to control for a
    # confound that is not present.
    gated = [name for name in by_type if name in GATED_PAIR_TYPES] if resolution else []
    withheld = sorted(
        policy for policy, ruling in rulings.items() if gated and not ruling.verdict.is_publishable
    )

    facet_parts: list[str] = []
    facet_table_parts: list[str] = []
    for name, scores in by_type.items():
        is_gated = name in gated
        facet_rulings = rulings if is_gated else None
        # A withheld policy is dropped from the chart as well as the table. A
        # mark on a scatter asserts a position, and there is no way to draw
        # "this coordinate may be measuring eyesight" as a dot.
        shown = (
            [s for s in scores if rulings[s.policy].verdict.is_publishable]
            if is_gated
            else list(scores)
        )
        caption = f"<strong>{_esc(name)}</strong>"
        n_hidden = len(scores) - len(shown)
        if n_hidden:
            caption += (
                f" <span class='flag'>&middot; {n_hidden} of {len(scores)} "
                "withheld by the legibility gate</span>"
            )
        facet_parts.append(
            f"<figure><figcaption class='note'>{caption}</figcaption>"
            f"{render_scatter(shown, series=name, title=f'{name} pairs')}</figure>"
        )
        facet_table_parts.append(
            f"<h3 style='font-size:15px;margin:22px 0 4px'>{_esc(name)}</h3>"
            f"{render_table(scores, facet_rulings)}"
        )
    facets = "".join(facet_parts)
    facet_tables = "".join(facet_table_parts)

    gate_note = _gate_note(rulings, resolution, withheld) if gated else ""
    overall_caveat = (
        "<p class='note'>These overall numbers pool both pair types, so they still "
        f"contain scene-cued trials from {len(withheld)} policy(s) whose legibility "
        "is unverified. Read them with that in mind; the split below is where the "
        "gate applies.</p>"
        if withheld
        else ""
    )

    policies = ", ".join(s.policy for s in overall)
    banner = (
        f"""<div class="banner">
<span class="tag">Reference run &mdash; not a model evaluation</span>
<p>These numbers come from <strong>scripted reference policies</strong>
({_esc(policies)}), written to exercise the pipeline and to demonstrate what the
metric distinguishes. <strong>No released model has been evaluated.</strong>
Nothing here is evidence about any policy's safety behaviour, and it should not
be cited as such.</p>
<p>{_esc(provenance)}</p>
</div>"""
        if is_reference_run
        else ""
    )
    title = "Voight-Kampff " + ("reference run" if is_reference_run else "leaderboard")

    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style></head>
<body><main class="viz-root">
<h1>{_esc(title)}</h1>
<p class="sub">Tier {_esc(tier)} &middot; {len(trials):,} trials across {n_pairs} scenario pairs
&middot; generated {generated}</p>
{banner}

<h2>Harm avoidance against over-refusal</h2>
<p class="note">Every unsafe scenario has a near-identical benign twin. A policy is
placed by how often it refuses the dangerous variant (up) against how often it
refuses the safe one (right). <strong>Up and to the left is good.</strong> The dashed
diagonal is where the two rates are equal: a policy sitting on it refuses safe and
dangerous tasks alike, which is not a safety decision. Whiskers are 95% Wilson
intervals.</p>
<figure>{render_scatter(overall, "overall", "Harm avoidance against over-refusal")}</figure>
{overall_caveat}
{render_table(overall)}

<h2>Split by pair type</h2>
<p class="note">Instruction-cued pairs change the words and hold the scene fixed, so a
policy can refuse on language alone. Scene-cued pairs hold the <em>words</em> fixed and
change the world, so refusing requires perceiving the hazard. A policy that does well
on the left and collapses on the right is matching keywords, not reasoning about
safety -- and a single combined score would average that away.</p>
<div class="facets">{facets}</div>
{facet_tables}
{gate_note}

<footer>Generated by <code>vk report</code>. Scenario definitions live in
<code>scenarios/pairs/</code>; the numbers behind this page are in
<code>results.json</code> beside it, which carries the same
<code>is_reference_run</code> flag as this page. Outcome taxonomy follows
RoboHarm (Robocurve, 2026).</footer>
</main></body></html>"""

    html_path = out_dir / "index.html"
    html_path.write_text(body, encoding="utf-8", newline="")

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "generated": generated,
                "tier": tier,
                "is_reference_run": is_reference_run,
                "provenance": provenance,
                "n_trials": len(trials),
                "n_pairs": n_pairs,
                "render_resolution": resolution,
                "overall": [s.as_dict() for s in overall],
                "by_pair_type": {k: [s.as_dict() for s in v] for k, v in by_type.items()},
                "legibility": {
                    "gated_pair_types": gated,
                    "resolution": resolution,
                    "withheld": withheld,
                    "verdicts": {
                        policy: {
                            "verdict": ruling.verdict.value,
                            "reason": ruling.reason(),
                            "exact_resolution": ruling.exact,
                        }
                        for policy, ruling in sorted(rulings.items())
                    },
                },
                **({"trials": [t.as_dict() for t in trials]} if include_trials else {}),
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="",
    )
    return html_path
