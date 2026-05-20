"""Central configuration for GestureFlow.

All tunable knobs live here so the rest of the codebase can be tested
with different configs and we never sprinkle magic numbers across modules.
Later we can hydrate this from a JSON/TOML file without touching call sites.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)
    # Mirror the frame horizontally so the user sees themselves naturally
    # ("selfie view"). This must happen BEFORE inference if we want gesture
    # semantics like "swipe right" to match the user's intent.
    mirror_frame: bool = True
    log_level: str = "INFO"
    window_title: str = "GestureFlow"


DEFAULT_CONFIG = Config()
