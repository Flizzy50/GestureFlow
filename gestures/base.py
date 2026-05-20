"""Abstract base and result types for gesture detectors.

The interface here is the contract every detector must honor. Keeping it
tiny — one method, one return type — pays off when we add Phase 4's state
machine, because the state machine only needs to know about `Detection`,
not about any specific gesture's internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vision.hand_tracker import Hand
from gestures.features import FingerExtension


@dataclass(frozen=True)
class FeatureBundle:
    """Per-frame, per-hand feature snapshot.

    Computed ONCE by the recognizer and passed to every detector. This
    matters: as we add more detectors, each shouldn't redundantly compute
    finger extensions or hand scale. It also guarantees every detector
    sees a consistent view of the same frame.
    """
    hand: Hand
    fingers: FingerExtension
    hand_scale: float


@dataclass(frozen=True)
class Detection:
    """A detector's verdict for one frame.

    confidence is in [0, 1]. metadata is for detector-specific extras
    (e.g., pinch distance, finger scores) that the HUD or state machine
    might want without re-running the detector.
    """
    name: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class GestureDetector(ABC):
    """Stateless per-frame gesture detector.

    Lifecycle: constructed once with its thresholds, called once per
    hand per frame. No frame counters, no last-seen state, no cooldowns
    — those belong in Phase 4's state machine, NOT here.

    Why this discipline matters: if temporal logic leaks into detectors,
    every detector ends up with its own private state and we can't reason
    about the system globally. Keeping detectors pure means Phase 4 can
    wrap ANY detector with debouncing, conflict resolution, or hysteresis
    without touching the detector code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable string id used by config bindings and HUD labels."""

    @abstractmethod
    def detect(self, features: FeatureBundle) -> Optional[Detection]:
        """Return a Detection when this gesture matches, else None."""
