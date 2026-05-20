"""MediaPipe Hands wrapper (Tasks API).

Why Tasks API and not the legacy mp.solutions.hands:
  The mediapipe wheel for Python 3.13+ does not ship the legacy
  mp.solutions submodule at all — only Image, ImageFormat, tasks.
  Tasks API is also Google's long-term direction, so this is the
  permanent move, not a stopgap.

What changed for callers:
  Nothing. HandTracker, Hand, HandLandmark, and HAND_CONNECTIONS keep
  the exact same interface. This is the payoff of wrapping MediaPipe
  behind our own domain types — only this file changes.

Coordinate convention (unchanged from MediaPipe):
  - x, y in [0, 1], normalized to image width/height.
  - z is relative depth vs the wrist; more negative = closer to camera.
  - Landmark indices follow MediaPipe's 21-point model:
      0 = wrist; 1-4 = thumb (CMC..TIP); 5-8 = index; 9-12 = middle;
      13-16 = ring; 17-20 = pinky.
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision

from config import HandTrackerConfig
from utils.logger import get_logger

log = get_logger(__name__)


# Standard MediaPipe 21-point hand skeleton. The Tasks API does NOT
# re-export this constant (it lived on mp.solutions.hands), so we own it.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # wrist -> finger bases
    (0, 1), (0, 5), (0, 17),
    # thumb
    (1, 2), (2, 3), (3, 4),
    # index
    (5, 6), (6, 7), (7, 8),
    # middle
    (9, 10), (10, 11), (11, 12),
    # ring
    (13, 14), (14, 15), (15, 16),
    # pinky
    (17, 18), (18, 19), (19, 20),
    # palm crossbars
    (5, 9), (9, 13), (13, 17),
)


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model(path: Path) -> Path:
    """Download the HandLandmarker .task if it isn't already on disk."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand_landmarker model from %s", _MODEL_URL)
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
    except Exception as e:
        raise RuntimeError(
            f"could not download MediaPipe model from {_MODEL_URL}. "
            f"Download it manually and place it at {path}."
        ) from e
    log.info("model downloaded (%.1f KB)", path.stat().st_size / 1024)
    return path


@dataclass(frozen=True)
class HandLandmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Hand:
    landmarks: Tuple[HandLandmark, ...]  # always length 21
    handedness: str                       # "Left" or "Right" (mirror-aware)
    handedness_score: float               # MediaPipe's confidence in L/R

    def landmark(self, idx: int) -> HandLandmark:
        return self.landmarks[idx]


class HandTracker:
    """Stateful MediaPipe HandLandmarker in VIDEO running mode.

    VIDEO mode is synchronous (returns the result inline) and requires
    monotonically increasing per-frame timestamps. We track our own
    monotonic clock and enforce strict-increase to survive the edge
    case where two consecutive frames fall in the same millisecond.
    """

    def __init__(self, config: HandTrackerConfig) -> None:
        self._cfg = config
        model_path = _ensure_model(Path(config.model_path))

        options = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = _mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    def process(self, bgr_frame: np.ndarray) -> List[Hand]:
        """Run hand detection on a BGR frame and return zero or more Hands."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int((time.monotonic() - self._t0) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.hand_landmarks:
            return []

        handedness_per_hand = result.handedness or []
        hands: List[Hand] = []
        for idx, lm_list in enumerate(result.hand_landmarks):
            if idx < len(handedness_per_hand) and handedness_per_hand[idx]:
                cat = handedness_per_hand[idx][0]
                label, score = cat.category_name, float(cat.score)
            else:
                label, score = "Unknown", 0.0
            landmarks = tuple(
                HandLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in lm_list
            )
            hands.append(
                Hand(landmarks=landmarks, handedness=label, handedness_score=score)
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
