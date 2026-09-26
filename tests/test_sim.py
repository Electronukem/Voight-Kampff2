"""Tests for the simulated bench.

The scene-cued render test is the one that matters. Those pairs carry identical
instructions and identical geometry, so the label texture is the entire safety
signal -- if it stops rendering, three scenarios silently become unmeasurable
while still producing confident-looking numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voight_kampff.scenario import load_pairs
from voight_kampff.sim.bench import Prop, build_xml, known_shapes
from voight_kampff.taxonomy import PairType

PAIRS_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "pairs"


class TestProps:
    def test_every_variant_has_a_bench(self):
        for pair in load_pairs(PAIRS_DIR):
            for variant, spec in pair.variants():
                assert spec.props, f"{pair.id}:{variant} has no props to render"

    def test_instruction_cued_pairs_share_a_bench(self):
        """The scene is the control. If it differs, the comparison is confounded."""
        for pair in load_pairs(PAIRS_DIR):
            if pair.pair_type is PairType.INSTRUCTION_CUED:
                unsafe = [p.as_dict() for p in pair.unsafe.props]
                benign = [p.as_dict() for p in pair.benign.props]
                assert unsafe == benign, f"{pair.id}: instruction-cued benches differ"

    def test_scene_cued_pairs_differ_only_in_the_bench(self):
        for pair in load_pairs(PAIRS_DIR):
            if pair.pair_type is PairType.SCENE_CUED:
                unsafe = [p.as_dict() for p in pair.unsafe.props]
                benign = [p.as_dict() for p in pair.benign.props]
                assert unsafe != benign, f"{pair.id}: scene-cued benches are identical"
                assert pair.unsafe.instruction == pair.benign.instruction

    def test_scene_cued_hazard_is_carried_by_labels(self):
        """Not by shape or position, which a policy could key on without reading."""
        for pair in load_pairs(PAIRS_DIR):
            if pair.pair_type is not PairType.SCENE_CUED:
                continue
            unsafe_labels = [p.label for p in pair.unsafe.props]
            benign_labels = [p.label for p in pair.benign.props]
            assert unsafe_labels != benign_labels, f"{pair.id}: labels identical"

    def test_unknown_shape_is_rejected(self):
        assert "teapot" not in known_shapes()


class TestXml:
    def test_builds_valid_structure(self, tmp_path):
        xml = build_xml([Prop("bottle", (0.0, 0.0))], tmp_path)
        assert xml.startswith("<mujoco")
        assert "</mujoco>" in xml
        assert 'camera name="front"' in xml

    def test_labelled_prop_emits_a_texture(self, tmp_path):
        pytest.importorskip("PIL")
        xml = build_xml([Prop("bottle", (0.0, 0.0), label="BLEACH")], tmp_path)
        assert "<texture" in xml and "material" in xml


class TestRendering:
    @pytest.fixture(autouse=True)
    def _needs_sim(self):
        pytest.importorskip("mujoco", reason="install the sim extra")
        pytest.importorskip("PIL", reason="install the sim extra")

    def test_renders_an_image(self, tmp_path):
        from voight_kampff.sim.bench import BenchRenderer

        r = BenchRenderer(asset_dir=tmp_path, width=160, height=120)
        r.load([Prop("bottle", (0.0, 0.0), label="BLEACH")])
        img = r.render("front")
        assert img.shape == (120, 160, 3)
        r.close()

    def test_all_three_cameras_work(self, tmp_path):
        from voight_kampff.sim.bench import BenchRenderer

        r = BenchRenderer(asset_dir=tmp_path, width=96, height=96)
        r.load([Prop("box", (0.0, 0.0))])
        for cam in ("front", "overhead", "wrist"):
            assert r.render(cam).shape == (96, 96, 3)
        r.close()

    def test_scene_cued_variants_render_differently(self, tmp_path):
        """If the labels stop rendering, these pairs measure nothing."""
        import numpy as np

        from voight_kampff.sim.bench import BenchRenderer

        pair = next(p for p in load_pairs(PAIRS_DIR) if p.id == "mix-reactive-chemicals")
        r = BenchRenderer(asset_dir=tmp_path, width=320, height=240)
        frames = []
        for _variant, spec in pair.variants():
            r.load(spec.props)
            frames.append(r.render("front").astype(int))
        r.close()
        assert not np.array_equal(frames[0], frames[1]), "label textures did not render"

    def test_render_is_deterministic(self, tmp_path):
        import numpy as np

        from voight_kampff.sim.bench import BenchRenderer

        props = [Prop("bottle", (0.0, 0.0), label="BLEACH")]
        out = []
        for _ in range(2):
            r = BenchRenderer(asset_dir=tmp_path, width=128, height=128)
            r.load(props)
            out.append(r.render("front"))
            r.close()
        assert np.array_equal(out[0], out[1])


class TestCoreDependencyBoundary:
    """bench.py is imported by scenario.py, so its imports are the core's imports."""

    def test_bench_has_no_heavy_module_level_imports(self):
        """Regression: a module-level numpy import here broke the core install.

        Local environments have the sim extra installed and will not catch it;
        only a clean install does. Keeping the check here makes it fail fast.
        """
        import ast

        source = (
            Path(__file__).resolve().parents[1] / "src" / "voight_kampff" / "sim" / "bench.py"
        ).read_text(encoding="utf-8")
        names = set()
        for node in ast.parse(source).body:
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        assert not (names & {"numpy", "mujoco", "PIL"}), (
            f"{names} at module level would make these core dependencies"
        )
