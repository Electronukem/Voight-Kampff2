"""Simulated benches, so a policy sees the scenario rather than a stand-in."""

from .bench import BenchRenderer, Prop, build_xml, known_shapes, label_texture

__all__ = ["BenchRenderer", "Prop", "build_xml", "known_shapes", "label_texture"]
