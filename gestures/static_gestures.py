"""Detectors for static (single-frame) gestures.

A detector matches on what the hand looks like *right now* — no movement
history. Anything that depends on motion (swipes) is a dynamic gesture
and lives elsewhere (Phase 4).
"""
from __future__ import annotations

from typing import Optional

from gestures.base import Detection, FeatureBundle, GestureDetector
from gestures.features import pinch_distance


class OpenPalmDetector(GestureDetector):
    """Open palm: all five fingers extended.

    Two-part check so we don't false-fire on poses with a high *average*
    extension but one obviously-curled finger:
      1) mean extension across all 5 fingers >= min_confidence
      2) the WEAKEST finger is still extended at least min_per_finger

    The second check rejects e.g. "four fingers up, thumb tucked" which
    would have a mean around 0.8 — high enough to pass (1) but clearly
    not an open palm.

    Confidence is the mean. Per-finger scores are surfaced in metadata
    for the HUD and for Phase 5's diagnostic overlay.
    """

    name = "open_palm"

    def __init__(
        self,
        min_confidence: float = 0.80,
        min_per_finger: float = 0.50,
    ) -> None:
        self._min_conf = min_confidence
        self._min_per = min_per_finger

    def detect(self, features: FeatureBundle) -> Optional[Detection]:
        scores = features.fingers.as_tuple
        mean = sum(scores) / 5.0
        if mean < self._min_conf:
            return None
        if min(scores) < self._min_per:
            return None
        return Detection(
            name=self.name,
            confidence=mean,
            metadata={"finger_scores": scores},
        )


class FistDetector(GestureDetector):
    """Closed fist: all four non-thumb fingers curled.

    We deliberately IGNORE the thumb. A natural fist has the thumb wrapped
    around the front of the fingers, which leaves the IP joint at maybe
    0.3-0.7 extension by our metric. Requiring thumb < threshold would
    reject most real fists. The four finger curls are the actual signal.

    Confidence: 1 - mean(non-thumb extension). Threshold filters by both
    mean curl AND requires the LEAST-curled finger to still be solidly
    closed, otherwise "two fingers up" with limp ring/pinky would sneak in.
    """

    name = "fist"

    def __init__(
        self,
        min_confidence: float = 0.75,
        max_per_finger: float = 0.40,
    ) -> None:
        self._min_conf = min_confidence
        self._max_per = max_per_finger

    def detect(self, features: FeatureBundle) -> Optional[Detection]:
        f = features.fingers
        non_thumb = (f.index, f.middle, f.ring, f.pinky)
        if max(non_thumb) > self._max_per:
            return None
        confidence = 1.0 - (sum(non_thumb) / 4.0)
        if confidence < self._min_conf:
            return None
        return Detection(
            name=self.name,
            confidence=confidence,
            metadata={"finger_scores": f.as_tuple},
        )


class PinchDetector(GestureDetector):
    """Pinch: thumb tip touching (or close to) index tip.

    The two failure modes a junior implementation would hit:
      1) Scale dependence — raw thumb-index distance shrinks with camera
         distance. We divide by hand_scale to get a depth-invariant ratio.
      2) Fist false-positive — in a fist, thumb tip is also near index
         (near the PIP joint, not the tip), but the index is curled.
         Requiring index_extension > min_index_extension rejects this.

    Confidence ramps continuously: 0.0 at `open_ratio` (fingers apart),
    1.0 at `closed_ratio` (touching). This smooth signal is useful for
    Phase 3's volume control — pinch tightness can map to a value.
    """

    name = "pinch"

    def __init__(
        self,
        closed_ratio: float = 0.20,
        open_ratio: float = 0.45,
        min_index_extension: float = 0.30,
        min_confidence: float = 0.50,
    ) -> None:
        if open_ratio <= closed_ratio:
            raise ValueError("open_ratio must be > closed_ratio")
        self._closed = closed_ratio
        self._open = open_ratio
        self._min_idx = min_index_extension
        self._min_conf = min_confidence

    def detect(self, features: FeatureBundle) -> Optional[Detection]:
        if features.fingers.index < self._min_idx:
            return None
        ratio = pinch_distance(features.hand)
        # Linear ramp: closed->1.0, open->0.0, clamp outside.
        confidence = (self._open - ratio) / (self._open - self._closed)
        if confidence < 0.0:
            confidence = 0.0
        elif confidence > 1.0:
            confidence = 1.0
        if confidence < self._min_conf:
            return None
        return Detection(
            name=self.name,
            confidence=confidence,
            metadata={"pinch_ratio": ratio},
        )


class TwoFingersDetector(GestureDetector):
    """Index and middle extended; ring and pinky curled; thumb don't-care.

    Confidence rewards both halves of the pose being clean:
        mean(idx, mid) * (1 - mean(ring, pinky))
    A "perfect" two-fingers pose scores 1.0. A half-hearted pose with
    floppy ring/pinky scores lower, so we don't over-fire.

    The thumb is don't-care because real two-finger poses vary —
    some people tuck it across the palm, some leave it out to the side.
    Requiring a thumb state would reject half of users' natural poses.
    """

    name = "two_fingers"

    def __init__(
        self,
        min_up: float = 0.65,
        max_down: float = 0.40,
        min_confidence: float = 0.45,
    ) -> None:
        self._min_up = min_up
        self._max_down = max_down
        self._min_conf = min_confidence

    def detect(self, features: FeatureBundle) -> Optional[Detection]:
        f = features.fingers
        if f.index < self._min_up or f.middle < self._min_up:
            return None
        if f.ring > self._max_down or f.pinky > self._max_down:
            return None
        up = (f.index + f.middle) / 2.0
        down_inv = 1.0 - (f.ring + f.pinky) / 2.0
        confidence = up * down_inv
        if confidence < self._min_conf:
            return None
        return Detection(
            name=self.name,
            confidence=confidence,
            metadata={"finger_scores": f.as_tuple},
        )
