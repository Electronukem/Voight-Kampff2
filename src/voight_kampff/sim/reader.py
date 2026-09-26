"""A policy that actually reads the bench, by template matching.

Every other reference policy in this repo either ignores the scene or cheats
with privileged state, so none of them can answer the question the legibility
probe exists to ask: at what camera resolution does label text stop carrying
information?

This one reads pixels. It renders the label named in the instruction using the
same function that textured the scene, then matches that reference against each
candidate region of the observed image and reaches for the best match. That is
not what a VLA does, and it is not meant to be -- it is a *ceiling*. It is
handed the exact ground-truth template, perfectly aligned in font and wording,
with no language understanding required. A vision-language model inferring the
text from scratch cannot do better than this on the same pixels.

So when its accuracy collapses at a given resolution, that is an
information-theoretic floor rather than a modelling failure: the text is no
longer there to be read, and any policy's scene-cued refusal rate at that
resolution is measuring something other than safety judgement.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

MAX_STEP = 0.06
_LABEL_RE = re.compile(r"labelled\s+(.+?)\s*$", re.IGNORECASE)


def extract_label(instruction: str) -> str | None:
    """Pull the target label out of a probe instruction."""
    match = _LABEL_RE.search(instruction.strip())
    return match.group(1).strip() if match else None


def _greyscale(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _normalise(patch: np.ndarray) -> np.ndarray:
    patch = patch - patch.mean()
    norm = np.linalg.norm(patch)
    return patch / norm if norm > 1e-9 else patch


def _resize(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize, to avoid a scipy/cv2 dependency."""
    h, w = shape
    rows = np.clip((np.arange(h) * arr.shape[0] / h).astype(int), 0, arr.shape[0] - 1)
    cols = np.clip((np.arange(w) * arr.shape[1] / w).astype(int), 0, arr.shape[1] - 1)
    return arr[np.ix_(rows, cols)]


# Label plate geometry, mirroring _geom_for in bench.py: the plate sits on the
# camera-facing side of the bottle, inset by its radius.
_BOTTLE_RADIUS = 0.045
_BOTTLE_HALF_H = 0.085
_PLATE_HALF_W = _BOTTLE_RADIUS * 0.85
_PLATE_HALF_H = min(_BOTTLE_HALF_H * 0.62, 0.055)
_PLATE_CENTRE_Z = _BOTTLE_HALF_H


def _crop_projected_label(
    grey: np.ndarray, world_x: float, width: int, height: int
) -> np.ndarray | None:
    """Crop the label plate of the candidate at ``world_x``.

    Uses camera projection rather than searching the image. That is deliberate:
    a real robot knows its own calibration, and the probe is measuring whether
    the *text* carries information, not whether a detector can find a bottle.
    Conflating the two would report a detection failure as illegibility.
    """
    from .bench import project_to_pixels

    plate_y = -_BOTTLE_RADIUS
    centre = project_to_pixels((world_x, plate_y, _PLATE_CENTRE_Z), width, height)
    edge = project_to_pixels((world_x + _PLATE_HALF_W, plate_y, _PLATE_CENTRE_Z), width, height)
    top = project_to_pixels((world_x, plate_y, _PLATE_CENTRE_Z + _PLATE_HALF_H), width, height)
    if centre is None or edge is None or top is None:
        return None

    half_w = max(2.0, abs(edge[0] - centre[0]))
    half_h = max(2.0, abs(top[1] - centre[1]))
    x0, x1 = round(centre[0] - half_w), round(centre[0] + half_w)
    y0, y1 = round(centre[1] - half_h), round(centre[1] + half_h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return grey[y0:y1, x0:x1]


class TemplateReaderPolicy:
    """Selects the labelled bottle by matching rendered text against the image.

    Each candidate's label plate is located by camera projection rather than by
    searching the image. Splitting the frame into equal bands does not work --
    the camera is perspective, so evenly spaced bottles do not land in evenly
    spaced thirds, and the far candidate falls outside its band entirely. That
    bug made the probe report chance performance at every resolution, which
    looked exactly like illegible text.
    """

    def __init__(
        self,
        *,
        n_choices: int = 3,
        resolution: int = 224,
        asset_dir: str | None = None,
        camera: str = "front",
    ):
        from inspect_robots.policy import PolicyConfig, PolicyInfo
        from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace

        self.n_choices = n_choices
        self.camera = camera
        self.asset_dir = Path(
            asset_dir or Path.home() / ".cache" / "voight-kampff" / "probe-templates"
        )
        self._choice: int | None = None
        self._scores: list[float] = []

        self.info = PolicyInfo(
            name="template-reader",
            action_space=Box(
                shape=(4,),
                low=np.array([-MAX_STEP, -MAX_STEP, -MAX_STEP, 0.0]),
                high=np.array([MAX_STEP, MAX_STEP, MAX_STEP, 1.0]),
                semantics=ActionSemantics(
                    control_mode="eef_delta_pos",
                    frame="world",
                    gripper="binary",
                    dim_labels=("dx", "dy", "dz", "grip"),
                ),
            ),
            # Reads the image and the effector, never the target. Asking for
            # target_pos would make the whole measurement meaningless.
            observation_space=ObservationSpace(
                cameras=(CameraSpec(name=camera, height=resolution, width=resolution),),
                state_keys=frozenset({"eef_pos"}),
            ),
        )
        self.config = PolicyConfig(action_horizon=1)

    def reset(self, scene: Any) -> None:
        self._choice = None
        self._scores = []

    def transcript(self) -> Any | None:
        if self._choice is None:
            return None
        return {"chose": self._choice, "match_scores": [round(s, 4) for s in self._scores]}

    def _reference(self, label: str) -> np.ndarray:
        from PIL import Image

        from .bench import label_texture

        path = label_texture(label, self.asset_dir)
        return _greyscale(np.asarray(Image.open(path).convert("RGB")))

    def _choose(self, observation: Any) -> int:
        label = extract_label(observation.instruction or "")
        images = observation.images or {}
        image = images.get(self.camera)
        if label is None or image is None:
            return 0

        grey = _greyscale(np.asarray(image))
        height, width = grey.shape
        reference = self._reference(label)

        scores = []
        for i in range(self.n_choices):
            plate = _crop_projected_label(grey, self._candidate_world_x(i), width, height)
            if plate is None or plate.size == 0:
                scores.append(-1.0)
                continue
            # Match at a fixed size so candidates at different depths -- the
            # bench is viewed in perspective -- are compared on equal terms.
            shape = (48, 48)
            a = _normalise(_resize(plate, shape))
            b = _normalise(_resize(reference, shape))
            scores.append(float((a * b).sum()))

        self._scores = scores
        return int(np.argmax(scores))

    def _candidate_world_x(self, slot: int) -> float:
        """Bench x of the given candidate, mirroring build_probe_scenes."""
        spacing = 0.22
        return -spacing * (self.n_choices - 1) / 2 + spacing * slot

    def act(self, observation: Any) -> Any:
        from inspect_robots.types import Action, ActionChunk

        if self._choice is None:
            self._choice = self._choose(observation)

        # Bench layout mirrors build_probe_scenes: evenly spaced, centred.
        spacing = 0.22
        offset = -spacing * (self.n_choices - 1) / 2
        target = np.array([offset + spacing * self._choice, 0.0, 0.08])

        eef = np.asarray(observation.state["eef_pos"], dtype=np.float64)
        delta = np.clip(target - eef, -MAX_STEP, MAX_STEP)
        return ActionChunk([Action(np.concatenate([delta, [1.0]]).astype(np.float64))])


class BlindReachPolicy:
    """Always reaches the leftmost bottle. The probe's floor control.

    Establishes that the probe cannot be passed without reading: a positional
    strategy has to land at chance. If this ever scores above chance the probe
    is leaking the answer through layout.
    """

    def __init__(self, *, n_choices: int = 3):
        from inspect_robots.policy import PolicyConfig, PolicyInfo
        from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace

        self.n_choices = n_choices
        self.info = PolicyInfo(
            name="blind-reach",
            action_space=Box(
                shape=(4,),
                low=np.array([-MAX_STEP, -MAX_STEP, -MAX_STEP, 0.0]),
                high=np.array([MAX_STEP, MAX_STEP, MAX_STEP, 1.0]),
                semantics=ActionSemantics(
                    control_mode="eef_delta_pos",
                    frame="world",
                    gripper="binary",
                    dim_labels=("dx", "dy", "dz", "grip"),
                ),
            ),
            observation_space=ObservationSpace(state_keys=frozenset({"eef_pos"})),
        )
        self.config = PolicyConfig(action_horizon=1)

    def reset(self, scene: Any) -> None:
        return None

    def act(self, observation: Any) -> Any:
        from inspect_robots.types import Action, ActionChunk

        spacing = 0.22
        target = np.array([-spacing * (self.n_choices - 1) / 2, 0.0, 0.08])
        eef = np.asarray(observation.state["eef_pos"], dtype=np.float64)
        delta = np.clip(target - eef, -MAX_STEP, MAX_STEP)
        return ActionChunk([Action(np.concatenate([delta, [1.0]]).astype(np.float64))])
