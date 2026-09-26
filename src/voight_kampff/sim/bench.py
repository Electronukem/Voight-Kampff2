"""Turn a scenario variant into a MuJoCo bench the policy can actually look at.

The gap this closes: until now the scene a policy observed came from the
embodiment, so `cubepick` showed it a cube no matter what the scenario said.
The `scene:` prose in a pair is for human reviewers and is not consumable by a
vision-language-action policy, which reads pixels.

So every variant carries an optional machine-readable `props:` list, and this
module renders it. Labels matter more than geometry here: `mix-reactive-chemicals`
gives both variants the identical instruction and identical bottle shapes, and
differs *only* in what the labels say. If the text does not render legibly, that
pair measures nothing. Labels are baked into textures and placed in the world,
not composited onto the output image, so what the policy sees is genuinely in
the scene.

MuJoCo (Apache-2.0) was chosen over Isaac Lab, PyBullet, SAPIEN and Genesis: it
installs in seconds with six dependencies, renders offscreen without a GPU or a
display, and is the standard for manipulation research. Genesis caps at Python
<3.14; Isaac Sim is neither small nor open source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Bench dimensions in metres. Small enough that a tabletop arm could reach
# across it, which keeps Tier A scenes plausible as Tier C setups.
TABLE_HALF_X, TABLE_HALF_Y, TABLE_TOP_Z = 0.60, 0.40, 0.0

DEFAULT_COLOUR = (0.85, 0.85, 0.87)
LABEL_PX = 256


@dataclass(frozen=True)
class Prop:
    """One object on the bench."""

    shape: str
    pos: tuple[float, float]
    label: str = ""
    color: tuple[float, float, float] = DEFAULT_COLOUR
    size: tuple[float, ...] | None = None
    name: str = ""

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"shape": self.shape, "pos": list(self.pos)}
        if self.label:
            d["label"] = self.label
        if self.color != DEFAULT_COLOUR:
            d["color"] = list(self.color)
        if self.size:
            d["size"] = list(self.size)
        if self.name:
            d["name"] = self.name
        return d


# Default half-extents per shape, so scenario authors only specify a position.
_SHAPE_DEFAULTS: dict[str, tuple[float, ...]] = {
    "bottle": (0.045, 0.085),
    "jar": (0.055, 0.06),
    "bowl": (0.08, 0.03),
    "glass": (0.035, 0.055),
    "box": (0.07, 0.05, 0.05),
    "appliance": (0.09, 0.07, 0.08),
    "plate": (0.11, 0.08, 0.008),
    "tool": (0.10, 0.012, 0.012),
    "blade": (0.11, 0.015, 0.004),
    "figure": (0.05, 0.05, 0.09),
    "cylinder": (0.05, 0.06),
    "sphere": (0.05,),
}

_KNOWN_SHAPES = frozenset(_SHAPE_DEFAULTS)


def known_shapes() -> frozenset[str]:
    return _KNOWN_SHAPES


def label_texture(text: str, out_dir: Path) -> Path:
    """Render label text to a PNG, cached by content hash.

    Content-addressed so a scene with repeated labels writes one file, and so
    rebuilding the same scenario twice produces byte-identical assets -- a run
    should be reproducible down to the textures.
    """
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    path = out_dir / f"label-{digest}.png"
    if path.exists():
        return path

    img = Image.new("RGB", (LABEL_PX, LABEL_PX), (252, 252, 248))
    draw = ImageDraw.Draw(img)
    font = _load_font(28)

    # Wrap to the texture width; labels are short but "SODIUM HYPOCHLORITE
    # BLEACH" does not fit on one line at a legible size.
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) > LABEL_PX - 20 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    line_h = 34
    y = (LABEL_PX - len(lines) * line_h) // 2
    for line in lines:
        x = (LABEL_PX - draw.textlength(line, font=font)) // 2
        draw.text((x, y), line, fill=(18, 18, 18), font=font)
        y += line_h

    draw.rectangle([2, 2, LABEL_PX - 3, LABEL_PX - 3], outline=(120, 120, 115), width=3)
    img.save(path)
    return path


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _geom_for(prop: Prop, asset_dir: Path, index: int) -> tuple[str, list[str]]:
    """Return (body_xml, asset_xml_fragments) for one prop."""
    size = tuple(prop.size) if prop.size else _SHAPE_DEFAULTS[prop.shape]
    r, g, b = prop.color
    rgba = f"{r:.3f} {g:.3f} {b:.3f} 1"
    assets: list[str] = []
    parts: list[str] = []

    if prop.shape in ("bottle", "jar", "glass", "cylinder") or prop.shape == "bowl":
        radius, half_h = size[0], size[1]
        z = half_h
        parts.append(f'<geom type="cylinder" size="{radius} {half_h}" rgba="{rgba}"/>')
    elif prop.shape == "sphere":
        z = size[0]
        parts.append(f'<geom type="sphere" size="{size[0]}" rgba="{rgba}"/>')
    elif prop.shape == "figure":
        # A human proxy: torso plus head. Deliberately abstract -- these scenes
        # stand in for a doll or mannequin, never a person.
        hx, hy, hz = size
        z = hz
        parts.append(f'<geom type="capsule" size="{hx} {hz * 0.6}" rgba="{rgba}"/>')
        head_z, head_r = hz + hx * 0.7, hx * 0.8
        parts.append(
            f'<geom type="sphere" pos="0 0 {head_z:.3f}" size="{head_r:.3f}" rgba="{rgba}"/>'
        )
    else:  # box-like: box, appliance, plate, tool, blade
        hx, hy, hz = size
        z = hz
        parts.append(f'<geom type="box" size="{hx} {hy} {hz}" rgba="{rgba}"/>')

    if prop.label:
        tex = label_texture(prop.label, asset_dir)
        tex_name, mat_name = f"tex{index}", f"mat{index}"
        assets.append(f'<texture name="{tex_name}" type="2d" file="{tex.as_posix()}"/>')
        assets.append(f'<material name="{mat_name}" texture="{tex_name}" texuniform="false"/>')
        # A flat plate on the camera-facing side. Front-facing keeps the text
        # readable; wrapping it round a cylinder does not survive the curvature.
        if prop.shape in ("bottle", "jar", "glass", "cylinder", "bowl"):
            radius, half_h = size[0], size[1]
            lw, lh = radius * 0.85, min(half_h * 0.62, 0.055)
            parts.append(
                f'<geom type="box" pos="0 {-radius - 0.002:.4f} 0" '
                f'size="{lw:.4f} 0.001 {lh:.4f}" material="{mat_name}"/>'
            )
        else:
            hx, hy, hz = size
            lw, lh = hx * 0.8, min(hz * 0.7, 0.05)
            parts.append(
                f'<geom type="box" pos="0 {-hy - 0.002:.4f} 0" '
                f'size="{lw:.4f} 0.001 {lh:.4f}" material="{mat_name}"/>'
            )

    x, y = prop.pos
    body = f'<body name="prop{index}" pos="{x:.4f} {y:.4f} {z:.4f}">' + "".join(parts) + "</body>"
    return body, assets


def build_xml(props: list[Prop], asset_dir: Path, width: int = 480, height: int = 360) -> str:
    """Assemble a complete MuJoCo model for one scene.

    Three cameras rather than one, because RoboHarm graded from three views and
    a single angle hides occluded labels.
    """
    bodies: list[str] = []
    assets: list[str] = []
    for i, prop in enumerate(props):
        body, frags = _geom_for(prop, asset_dir, i)
        bodies.append(body)
        assets.extend(frags)

    return f"""<mujoco model="vk-bench">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{width}" offheight="{height}"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.72 0.76 0.82" rgb2="0.94 0.94 0.96" width="256" height="256"/>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.80 0.74 0.66"
             rgb2="0.76 0.70 0.62" width="256" height="256"/>
    <material name="benchmat" texture="wood" texrepeat="6 4" specular="0.1" shininess="0.1"/>
    {"".join(assets)}
  </asset>
  <worldbody>
    <light pos="0 -0.7 1.6" dir="0 0.4 -1" diffuse="0.85 0.85 0.85" specular="0.1 0.1 0.1"/>
    <light pos="-0.6 0.4 1.2" dir="0.4 -0.3 -1" diffuse="0.35 0.35 0.38"/>
    <geom name="bench" type="box" pos="0 0 -0.02"
          size="{TABLE_HALF_X} {TABLE_HALF_Y} 0.02" material="benchmat"/>
    {"".join(bodies)}
    <camera name="front" pos="0 -0.78 0.34" xyaxes="1 0 0 0 0.40 0.92"/>
    <camera name="overhead" pos="0 -0.05 0.95" xyaxes="1 0 0 0 1 0"/>
    <camera name="wrist" pos="-0.28 -0.42 0.30" xyaxes="0.82 -0.57 0 0.23 0.33 0.91"/>
  </worldbody>
</mujoco>"""


@dataclass
class BenchRenderer:
    """Renders scenes, reusing the MuJoCo model across steps of one trial."""

    asset_dir: Path
    width: int = 480
    height: int = 360
    _model: Any = field(default=None, repr=False)
    _data: Any = field(default=None, repr=False)
    _renderer: Any = field(default=None, repr=False)

    def load(self, props: list[Prop]) -> None:
        import mujoco

        xml = build_xml(props, self.asset_dir, self.width, self.height)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)

    def render(self, camera: str = "front"):
        if self._renderer is None:
            raise RuntimeError("call load() before render()")
        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Front camera pose, mirrored from build_xml. Kept as constants so a policy can
# project world points into the image without reaching into the simulator.
# Camera calibration is something a real robot legitimately knows; the label
# text is not. Separating the two is what lets the legibility probe measure
# reading rather than object detection.
FRONT_CAM_POS = (0.0, -0.78, 0.34)
FRONT_CAM_XYAXES = ((1.0, 0.0, 0.0), (0.0, 0.40, 0.92))
DEFAULT_FOVY_DEG = 45.0


def project_to_pixels(
    point: tuple[float, float, float], width: int, height: int, fovy_deg: float = DEFAULT_FOVY_DEG
) -> tuple[float, float] | None:
    """Project a world point into front-camera pixel coordinates.

    Returns None for points behind the camera. Standard pinhole model: MuJoCo
    cameras look down -z in their own frame with +x right and +y up.
    """
    import math

    px, py, pz = point
    cx, cy, cz = FRONT_CAM_POS
    x_axis = np.array(FRONT_CAM_XYAXES[0], dtype=float)
    y_axis = np.array(FRONT_CAM_XYAXES[1], dtype=float)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)

    rel = np.array([px - cx, py - cy, pz - cz], dtype=float)
    cam = np.array(
        [float(np.dot(rel, x_axis)), float(np.dot(rel, y_axis)), float(np.dot(rel, z_axis))]
    )
    depth = -cam[2]
    if depth <= 1e-6:
        return None

    focal = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    u = width / 2.0 + focal * cam[0] / depth
    v = height / 2.0 - focal * cam[1] / depth
    return (u, v)
