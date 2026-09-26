"""Tests for the presentation figures.

The argument figure is the one that can go quietly wrong: its caption claims
that refusing everything scores zero, and if the policies on the chart do not
include a refuse-everything baseline, the caption asserts something the image
does not show.
"""

from __future__ import annotations

import pytest

from voight_kampff.report.figures import find_browser


class TestArgumentFigure:
    @pytest.fixture(autouse=True)
    def _needs_sim(self):
        pytest.importorskip("mujoco", reason="install the sim extra")
        pytest.importorskip("PIL", reason="install the sim extra")

    def test_reference_trials_include_the_degenerate_corners(self):
        """Without these the figure's caption is unsupported by its own data."""
        from voight_kampff.report.figures import reference_trials

        policies = {t.policy for t in reference_trials()}
        assert "always-refuse" in policies
        assert "never-refuse" in policies

    def test_always_refuse_lands_on_the_diagonal(self):
        """The claim the figure is making, checked rather than captioned."""
        from voight_kampff.report.figures import reference_trials
        from voight_kampff.scoring import score_all

        scores = {s.policy: s for s in score_all(reference_trials())}
        always = scores["always-refuse"]
        assert always.harm_avoidance.point == pytest.approx(1.0)
        assert always.discrimination == pytest.approx(0.0)

    def test_scene_cued_figure_renders(self, tmp_path):
        from voight_kampff.report.figures import scene_cued_figure

        path = scene_cued_figure("mix-reactive-chemicals", tmp_path, width=160, height=120)
        assert path is not None and path.exists()

    def test_unknown_pair_returns_none(self, tmp_path):
        from voight_kampff.report.figures import scene_cued_figure

        assert scene_cued_figure("no-such-pair", tmp_path) is None

    def test_legibility_figure_spans_resolutions(self, tmp_path):
        from PIL import Image

        from voight_kampff.report.figures import legibility_figure

        path = legibility_figure("mix-reactive-chemicals", tmp_path, resolutions=(64, 128))
        assert path is not None
        width, height = Image.open(path).size
        assert width > height  # tiles laid side by side


class TestBrowserDiscovery:
    def test_returns_a_path_or_none(self):
        """Absence is reported, not raised: figures degrade rather than fail."""
        browser = find_browser()
        assert browser is None or isinstance(browser, str)
