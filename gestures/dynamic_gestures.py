"""Detectors for dynamic (motion-based) gestures.

Unlike static_gestures.py, these read FeatureBundle.motion — populated
by GestureRecognizer from its injected MotionTracker. From the detector's
own perspective they're still stateless: 'given a motion snapshot, does
this gesture match right now?'

Sign convention reminder:
  The main loop mirrors the frame BEFORE inference. So in the coordinates
  the recognizer sees, moving the hand to the USER'S left makes x
  decrease. SwipeLeftDetector requires dx < 0.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from gestures.base import Detection, FeatureBundle, GestureDetector


class _SwipeDetectorBase(GestureDetector):
    """Shared swipe logic. Subclasses set the direction sign."""

    def __init__(
        self,
        min_displacement: float = 0.15,
        min_velocity: float = 0.4,
        max_vertical_ratio: float = 0.7,
        min_samples: int = 3,
        min_open_fingers: int = 2,
        open_finger_threshold: float = 0.4,
        min_confidence: float = 0.4,
        velocity_for_full_confidence: float = 1.0,
    ) -> None:
        # Defaults are tuned empirically for ~17 FPS + relaxed hand poses.
        # Real human swipes have partly-curled fingers, especially during
        # motion (MediaPipe predicts curled-looking landmarks on blurred
        # fingers). Strict open-palm gating rejects most real swipes; the
        # velocity + displacement filters carry most of the false-positive
        # protection.
        if min_displacement <= 0:
            raise ValueError("min_displacement must be positive")
        if min_velocity <= 0:
            raise ValueError("min_velocity must be positive")
        self._min_disp = min_displacement
        self._min_vel = min_velocity
        self._max_vert = max_vertical_ratio
        self._min_samples = min_samples
        self._min_open = min_open_fingers
        self._open_thresh = open_finger_threshold
        self._min_conf = min_confidence
        self._vel_full = velocity_for_full_confidence

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def _sign(self) -> int:
        """+1 for rightward swipe, -1 for leftward swipe."""

    def detect(self, features: FeatureBundle) -> Optional[Detection]:
        motion = features.motion
        if motion is None or motion.sample_count < self._min_samples:
            return None

        # Open-hand gate (loose). Prevents fist-and-throw from registering
        # but tolerates the partly-curled hand of a real-world swipe.
        # Pinch+horizontal motion is fine — VolumeAction is already engaged
        # and consuming the gesture.
        if features.fingers.count_extended(threshold=self._open_thresh) < self._min_open:
            return None

        # Direction filter: dx must point the right way and be large enough.
        signed_dx = motion.dx * self._sign  # positive if in the right direction
        if signed_dx < self._min_disp:
            return None

        signed_vx = motion.velocity_x * self._sign
        if signed_vx < self._min_vel:
            return None

        # Reject mostly-vertical motion — that's a scroll setup or a wave.
        if abs(motion.dy) > self._max_vert * abs(motion.dx):
            return None

        # Confidence ramps with velocity, clamped to 1.0.
        confidence = signed_vx / self._vel_full
        if confidence > 1.0:
            confidence = 1.0
        if confidence < self._min_conf:
            return None

        return Detection(
            name=self.name,
            confidence=confidence,
            metadata={
                "dx": motion.dx,
                "dy": motion.dy,
                "dt": motion.dt,
                "velocity_x": motion.velocity_x,
                "sample_count": motion.sample_count,
            },
        )


class SwipeLeftDetector(_SwipeDetectorBase):
    """Swipe leftward — open hand, horizontal motion, dx < 0."""
    name = "swipe_left"

    @property
    def _sign(self) -> int:
        return -1


class SwipeRightDetector(_SwipeDetectorBase):
    """Swipe rightward — open hand, horizontal motion, dx > 0."""
    name = "swipe_right"

    @property
    def _sign(self) -> int:
        return 1
