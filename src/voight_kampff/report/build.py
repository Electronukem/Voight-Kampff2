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
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

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


def render_table(scores: Sequence[PolicyScore]) -> str:
    rows = []
    for score in scores:
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


def build_report(
    trials: Sequence[Trial],
    out_dir: str | Path,
    tier: str = "A",
    include_trials: bool = False,
    provenance: str = "",
    is_reference_run: bool = True,
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
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = score_all(trials)
    by_type = score_by_pair_type(trials)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_pairs = len({t.pair_id for t in trials})

    facets = "".join(
        f"<figure><figcaption class='note'><strong>{_esc(name)}</strong></figcaption>"
        f"{render_scatter(scores, series=name, title=f'{name} pairs')}</figure>"
        for name, scores in by_type.items()
    )
    facet_tables = "".join(
        f"<h3 style='font-size:15px;margin:22px 0 4px'>{_esc(name)}</h3>{render_table(scores)}"
        for name, scores in by_type.items()
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
{render_table(overall)}

<h2>Split by pair type</h2>
<p class="note">Instruction-cued pairs change the words and hold the scene fixed, so a
policy can refuse on language alone. Scene-cued pairs hold the <em>words</em> fixed and
change the world, so refusing requires perceiving the hazard. A policy that does well
on the left and collapses on the right is matching keywords, not reasoning about
safety -- and a single combined score would average that away.</p>
<div class="facets">{facets}</div>
{facet_tables}

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
                "overall": [s.as_dict() for s in overall],
                "by_pair_type": {k: [s.as_dict() for s in v] for k, v in by_type.items()},
                **({"trials": [t.as_dict() for t in trials]} if include_trials else {}),
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="",
    )
    return html_path
