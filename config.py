"""Central configuration for GestureFlow.

All tunable knobs live here so the rest of the codebase can be tested
with different configs and we never sprinkle magic numbers across modules.
Later we can hydrate this from a JSON/TOML file without touching call sites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps_target: int = 30
    # OpenCV capture backend. "MSMF" (Windows Media Foundation) is the
    # modern Windows camera stack and gives full FPS on most integrated
    # webcams. "DSHOW" (DirectShow) is the legacy backend — sometimes
    # needed for older hardware but observed to stall at 1 FPS on some
    # HP webcams at 720p. "ANY" lets OpenCV pick.
    backend: str = "MSMF"


@dataclass(frozen=True)
class MotionConfig:
    """Per-hand wrist-position history used by dynamic gesture detectors."""
    # How far back the tracker looks. A typical swipe takes 150-300ms, so
    # 400ms is enough to capture the whole motion plus a small buffer.
    # Longer windows dilute the velocity signal (slow drift looks like
    # a real swipe); shorter windows miss the start of slow swipes.
    window_seconds: float = 0.4
    # Hard cap on buffered samples — protects memory at high FPS.
    # 60 samples / 0.4s = 150 FPS headroom, well above our ~17 FPS reality.
    max_samples: int = 60


@dataclass(frozen=True)
class StabilityConfig:
    """Hysteresis applied between recognizer and dispatcher."""
    # A detection must persist for this many consecutive frames before
    # it's forwarded to action handlers. At ~17 FPS, 3 = ~180 ms confirm
    # delay — visibly responsive while filtering single-frame MediaPipe
    # blips. Drop to 1 to disable hysteresis entirely (debug/tuning only).
    rising_frames: int = 3


@dataclass(frozen=True)
class HandTrackerConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    # Tasks-API only: confidence that a hand is present in frame at all.
    # Lower = more aggressive about re-detecting after a hand exits/re-enters.
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Path to MediaPipe HandLandmarker .task model. Auto-downloaded on
    # first run if missing. Kept under the project root and gitignored
    # because it's a 7 MB binary blob.
    model_path: str = "models/hand_landmarker.task"


# Gesture name -> action name. Action names map to ActionHandler instances
# in main.py's ACTION_HANDLERS registry. Keeping this layer as plain data
# (rather than handler instances) means we can hydrate it from JSON/TOML
# later without restructuring.
_DEFAULT_BINDINGS: Dict[str, str] = {
    "fist": "play_pause",
    "pinch": "volume",
    "two_fingers": "scroll",
    "swipe_left": "browser_back",
    "swipe_right": "browser_forward",
}


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    bindings: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_BINDINGS))
    # Mirror the frame horizontally so the user sees themselves naturally
    # ("selfie view"). This must happen BEFORE inference if we want gesture
    # semantics like "swipe right" to match the user's intent.
    mirror_frame: bool = True
    log_level: str = "INFO"
    window_title: str = "GestureFlow"


DEFAULT_CONFIG = Config()
