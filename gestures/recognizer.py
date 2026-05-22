"""Orchestrator: compute features once, ask every detector, rank results."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from vision.hand_tracker import Hand
from gestures.base import Detection, FeatureBundle, GestureDetector
from gestures.features import finger_extensions, hand_scale
from gestures.motion import MotionTracker


class GestureRecognizer:
    """Run a fixed set of detectors over a single hand.

    Stateless w.r.t. detection logic. Stateful temporal context (motion
    history) is held by an injected MotionTracker — the recognizer only
    reads snapshots from it, never owns the buffers.

    Hysteresis, cooldowns, and conflict resolution belong elsewhere
    (StabilityFilter, action handlers' debounce policies).
    """

    def __init__(
        self,
        detectors: Sequence[GestureDetector],
        motion_tracker: Optional[MotionTracker] = None,
    ) -> None:
        if not detectors:
            raise ValueError("GestureRecognizer requires at least one detector")
        self._detectors: Tuple[GestureDetector, ...] = tuple(detectors)
        self._motion = motion_tracker

    @property
    def detector_names(self) -> Tuple[str, ...]:
        return tuple(d.name for d in self._detectors)

    def process(self, hand: Hand) -> List[Detection]:
        """Return all detections matching this hand, highest confidence first.

        Note: MotionTracker.update() must be called separately by the
        caller (typically main.py) — we don't pass `now` through here
        because doing so would break every existing caller's signature.
        """
        motion = self._motion.snapshot(hand.handedness) if self._motion else None
        features = FeatureBundle(
            hand=hand,
            fingers=finger_extensions(hand),
            hand_scale=hand_scale(hand),
            motion=motion,
        )
        detections: List[Detection] = []
        for detector in self._detectors:
            d = detector.detect(features)
            if d is not None:
                detections.append(d)
        detections.sort(key=lambda x: -x.confidence)
        return detections
