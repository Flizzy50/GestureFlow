"""Per-hand motion history for dynamic gesture detection.

Static detectors look at a single frame. Dynamic detectors (swipes,
eventually rotations) need to know what the hand was DOING over a short
recent window. Rather than scattering ring buffers across every dynamic
detector, we centralize that state in a MotionTracker.

Composition over inheritance: the tracker is a separate object that's
injected into the recognizer. The recognizer asks the tracker for a
read-only snapshot once per frame and bundles it into FeatureBundle for
detectors to consume. Tracker stays independently testable; detectors
stay stateless.

Coordinates: we track wrist position in normalized image space (the same
[0, 1] coordinates MediaPipe uses for everything). After the main loop's
horizontal mirror, x increasing means the hand moved to the USER'S
right; x decreasing means the user's left.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from gestures.features import LM
from vision.hand_tracker import Hand


@dataclass(frozen=True)
class MotionSample:
    """A single timestamped wrist position."""
    t: float
    x: float
    y: float


@dataclass(frozen=True)
class MotionSnapshot:
    """Derived motion features over a window of samples.

    dx, dy:        net displacement (last - first) in normalized coords.
    dt:            time spanned by the analyzed samples (seconds).
    sample_count:  number of samples in the snapshot. Detectors should
                   require a minimum count to avoid firing on stutter.
    """
    dx: float
    dy: float
    dt: float
    sample_count: int

    @property
    def velocity_x(self) -> float:
        return self.dx / self.dt if self.dt > 0 else 0.0

    @property
    def velocity_y(self) -> float:
        return self.dy / self.dt if self.dt > 0 else 0.0


class MotionTracker:
    """Ring buffers of recent wrist positions, keyed by handedness.

    Why keyed by handedness, not by 'first hand seen': MediaPipe tags each
    detection with Left/Right, which is stable across frames for the same
    physical hand. Tracking by label keeps the two hands' motion separate
    so a swipe from one doesn't pollute the other's buffer.

    'Unknown' handedness (rare) falls into its own bucket. Not perfect
    but acceptable — those events should be vanishingly rare in practice.
    """

    def __init__(self, window_seconds: float = 0.4, max_samples: int = 60) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_samples < 2:
            raise ValueError("max_samples must be >= 2")
        self._window = window_seconds
        self._max_samples = max_samples
        self._buffers: Dict[str, Deque[MotionSample]] = {}

    @property
    def window_seconds(self) -> float:
        return self._window

    def update(self, hand: Hand, now: float) -> None:
        """Record this hand's current wrist position at time `now`."""
        key = hand.handedness or "Unknown"
        buf = self._buffers.get(key)
        if buf is None:
            buf = deque(maxlen=self._max_samples)
            self._buffers[key] = buf
        wrist = hand.landmark(LM.WRIST)
        buf.append(MotionSample(t=now, x=wrist.x, y=wrist.y))
        # Drop samples older than the window. maxlen handles overflow by
        # length; this handles overflow by age. Both bounds matter — at
        # low FPS, age expires first; at high FPS, length expires first.
        cutoff = now - self._window
        while buf and buf[0].t < cutoff:
            buf.popleft()

    def snapshot(self, handedness: str) -> Optional[MotionSnapshot]:
        """Return motion features for the named hand, or None if there
        isn't enough data (fewer than 2 samples or dt <= 0)."""
        buf = self._buffers.get(handedness)
        if buf is None or len(buf) < 2:
            return None
        first = buf[0]
        last = buf[-1]
        dt = last.t - first.t
        if dt <= 0:
            return None
        return MotionSnapshot(
            dx=last.x - first.x,
            dy=last.y - first.y,
            dt=dt,
            sample_count=len(buf),
        )

    def reset(self) -> None:
        """Discard all buffered samples for every hand. Useful for tests
        and on activation toggles."""
        self._buffers.clear()
