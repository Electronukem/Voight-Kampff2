"""Standalone figures, sized for slides.

The leaderboard page is the wrong thing to present. It reports scripted
policies, and an audience that knows the field will ask which models were
evaluated -- a question with no good answer yet. Led with, it reads as a
result. Shown after the method, it reads as evidence the method is built,
which is what it is.

So these are the figures that carry the argument instead:

    two-axis        why a one-axis safety leaderboard is won by `return REFUSE`
    scene-cued      two benches differing only in what two labels say
    legibility      the resolution at which that text stops being readable

Each is written as a PNG at presentation resolution, because slide software
handles PNGs and does not handle a local web page reliably in front of an
audience.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# Where headless Chromium-family browsers usually live on Windows and macOS.
# Used to rasterise the SVG chart, which is otherwise only renderable in a page.
_BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
)


def find_browser() -> str | None:
    for candidate in _BROWSERS:
        if Path(candidate).exists():
            return candidate
    return None


def _shoot(browser: str, html: Path, png: Path, width: int, height: int) -> bool:
    result = subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--screenshot={png}",
            html.resolve().as_uri(),
        ],
        capture_output=True,
        timeout=120,
    )
    del result
    return png.exists()


def reference_trials() -> Any:
    """Run the scripted reference policies in process.

    The argument figure needs the degenerate corners on it. A run of real
    policies will not contain a refuse-everything baseline, so a scatter built
    from one cannot show that refusing everything scores zero -- which is the
    entire claim the figure exists to make. These stubs are instant, need no
    simulator, and are labelled as scripted wherever they appear.
    """
    from ..harness.base import run_many
    from ..harness.mock import reference_policies
    from ..scenario import load_pairs

    return run_many(reference_policies(), load_pairs("scenarios/pairs"), trials_per_variant=20)


def two_axis_figure(trials: Any, out_dir: str | Path, width: int = 1200) -> Path | None:
    """The scatter on its own, with the caption that makes it an argument."""
    from ..scoring import score_all
    from .build import _CSS, render_scatter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = score_all(trials)

    html = out_dir / "_two-axis.html"
    html.write_text(
        f"""<!DOCTYPE html><html lang="en" data-theme="light"><head><meta charset="utf-8">
<style>{_CSS}
body {{ background: #fff; }}
.viz-root {{ max-width: 980px; padding: 28px 24px; }}
h2 {{ margin: 0 0 6px; font-size: 22px; }}
figure {{ background: #fff; padding: 8px 0 0; }}
</style></head><body><main class="viz-root">
<h2>Rank on refusal alone, and <code>return REFUSE</code> wins</h2>
<p class="note">Every unsafe scenario has a near-identical benign twin. Up is
refusing the dangerous variant; right is refusing the safe one. The dashed
diagonal is where those two rates are equal &mdash; a policy on that line
refuses safe and dangerous tasks at the same rate, which is not a safety
decision at all. <strong>always-refuse sits top right with 100% harm avoidance
and scores zero.</strong> Scripted reference policies; no released model has
been evaluated.</p>
<figure>{render_scatter(scores, "overall", "Harm avoidance against over-refusal")}</figure>
</main></body></html>""",
        encoding="utf-8",
        newline="",
    )

    browser = find_browser()
    if browser is None:
        return None
    png = out_dir / "two-axis.png"
    # Sized to the content: a taller window leaves white space that has to
    # be cropped out before the figure can go on a slide.
    ok = _shoot(browser, html, png, width, int(width * 0.60))
    html.unlink(missing_ok=True)
    return png if ok else None


def scene_cued_figure(
    pair_id: str, out_dir: str | Path, width: int = 720, height: int = 520
) -> Path | None:
    """Both variants of one scene-cued pair, side by side.

    The most persuasive single image this project produces: two benches a
    viewer can see are identical, where the entire difference in what a robot
    should do is a few hundred pixels of text.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    from ..scenario import load_pairs
    from ..sim.bench import BenchRenderer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {p.id: p for p in load_pairs("scenarios/pairs")}
    if pair_id not in pairs:
        return None

    renderer = BenchRenderer(asset_dir=out_dir / "_assets", width=width, height=height)
    frames = []
    for _variant, spec in pairs[pair_id].variants():
        if not spec.props:
            renderer.close()
            return None
        renderer.load(spec.props)
        frames.append(renderer.render("front"))
    renderer.close()

    gap = np.full((height, 8, 3), 255, dtype=np.uint8)
    combined = np.concatenate([frames[0], gap, frames[1]], axis=1)
    png = out_dir / f"scene-cued-{pair_id}.png"
    Image.fromarray(combined).save(png)
    return png


def legibility_figure(
    pair_id: str, out_dir: str | Path, resolutions: tuple[int, ...] = (112, 224, 448)
) -> Path | None:
    """The same bench at several camera resolutions, upscaled to match.

    Shows the confound directly rather than asserting it: at 112px the label
    is gone, and a policy failing there has not made a safety mistake.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    from ..scenario import load_pairs
    from ..sim.bench import BenchRenderer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {p.id: p for p in load_pairs("scenarios/pairs")}
    if pair_id not in pairs:
        return None
    props = pairs[pair_id].unsafe.props
    if not props:
        return None

    target = max(resolutions)
    tiles = []
    for resolution in resolutions:
        renderer = BenchRenderer(asset_dir=out_dir / "_assets", width=resolution, height=resolution)
        renderer.load(props)
        frame = renderer.render("front")
        renderer.close()
        # Nearest-neighbour upscale so the reader sees the information that was
        # captured, not an interpolation inventing detail that was never there.
        tiles.append(np.array(Image.fromarray(frame).resize((target, target), Image.NEAREST)))
        tiles.append(np.full((target, 8, 3), 255, dtype=np.uint8))

    png = out_dir / "legibility.png"
    Image.fromarray(np.concatenate(tiles[:-1], axis=1)).save(png)
    return png


def build_figures(trials: Any, out_dir: str | Path, scene_pair: str) -> dict[str, Any]:
    """Write every figure, reporting which ones could be produced."""
    out_dir = Path(out_dir)
    made: dict[str, Any] = {}
    # The argument figure always uses the reference stubs, because it has to
    # show the degenerate corners. `trials` drives the others.
    made["two_axis"] = two_axis_figure(reference_trials(), out_dir)
    made["scene_cued"] = scene_cued_figure(scene_pair, out_dir)
    made["legibility"] = legibility_figure(scene_pair, out_dir)

    manifest = out_dir / "figures.json"
    manifest.write_text(
        json.dumps({k: (str(v) if v else None) for k, v in made.items()}, indent=2),
        encoding="utf-8",
        newline="",
    )
    return made
