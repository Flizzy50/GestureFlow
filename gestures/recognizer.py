"""Orchestrator: compute features once, ask every detector, rank results."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from vision.hand_tracker import Hand
from gestures.base import Detection, FeatureBundle, GestureDetector
from gestures.features import finger_extensions, hand_scale


class GestureRecognizer:
    """Run a fixed set of detectors over a single hand.

    The recognizer is intentionally NOT a state machine. It returns every
    matching detection on every frame, sorted by confidence. Phase 4's
    state machine will sit on top and add hysteresis, cooldowns, and
    conflict resolution — but those concerns don't pollute this layer.
    """

    def __init__(self, detectors: Sequence[GestureDetector]) -> None:
        if not detectors:
            raise ValueError("GestureRecognizer requires at least one detector")
        self._detectors: Tuple[GestureDetector, ...] = tuple(detectors)

    @property
    def detector_names(self) -> Tuple[str, ...]:
        return tuple(d.name for d in self._detectors)

    def process(self, hand: Hand) -> List[Detection]:
        """Return all detections matching this hand, highest confidence first."""
        features = FeatureBundle(
            hand=hand,
            fingers=finger_extensions(hand),
            hand_scale=hand_scale(hand),
        )
        detections: List[Detection] = []
        for detector in self._detectors:
            d = detector.detect(features)
            if d is not None:
                detections.append(d)
        detections.sort(key=lambda x: -x.confidence)
        return detections
