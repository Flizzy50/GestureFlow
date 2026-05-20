"""Per-gesture stability filtering between the recognizer and the dispatcher.

The recognizer is stateless and emits a fresh verdict every frame. That's
the right shape for testability and clean composition — but it means
single-frame noise from MediaPipe (a finger blurs during fast motion, a
landmark briefly mispredicts) shows up as a spurious detection that
would immediately fire a one-shot action.

StabilityFilter is the rising-edge debouncer that sits in the middle.
Each gesture name accumulates a 'streak' of consecutive frames in which
it was detected. Only after the streak hits `rising_frames` does the
detection become 'stable' and get forwarded downstream.

Falling-edge hysteresis (keeping a gesture 'live' for M frames after it
disappears) is deliberately NOT included here yet — continuous handlers
need per-frame data once engaged, and stale data would degrade their UX.
If real-world testing surfaces falling-edge jitter, add it then.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from gestures.base import Detection


class StabilityFilter:
    """Rising-edge hysteresis on a stream of per-frame detections."""

    def __init__(self, rising_frames: int = 3) -> None:
        if rising_frames < 1:
            raise ValueError("rising_frames must be >= 1")
        self._rising = rising_frames
        self._streaks: Dict[str, int] = {}

    @property
    def rising_threshold(self) -> int:
        return self._rising

    @property
    def streaks(self) -> Mapping[str, int]:
        """Current consecutive-detection counts per gesture name.

        Exposed so the HUD can show 'pending' indicators ('open_palm 2/3')
        for gestures that haven't yet crossed the stability threshold.
        Returns a copy so callers can't mutate internal state.
        """
        return dict(self._streaks)

    def update(self, detections: Sequence[Detection]) -> List[Detection]:
        """Tick the filter for one frame.

        Args:
            detections: every raw detection from the recognizer for this
                frame, possibly multiple per gesture name across hands.

        Returns:
            Only those detections whose gesture has been continuously
            detected for at least `rising_frames` frames. Sorted by
            confidence descending so downstream stays deterministic.
        """
        # Collapse to best-per-gesture-name (multi-hand: take the higher
        # confidence). Mirrors what the dispatcher does — accepted
        # duplication, see module docstring discussion in main.py.
        best: Dict[str, Detection] = {}
        for det in detections:
            existing = best.get(det.name)
            if existing is None or det.confidence > existing.confidence:
                best[det.name] = det

        stable: List[Detection] = []

        # Increment streaks for present gestures; reset for absent ones.
        # Iterate over the UNION so absent-this-frame gestures get cleared.
        all_known = set(self._streaks) | set(best)
        for name in all_known:
            if name in best:
                streak = self._streaks.get(name, 0) + 1
                self._streaks[name] = streak
                if streak >= self._rising:
                    stable.append(best[name])
            else:
                # Gesture absent this frame — reset its counter.
                self._streaks.pop(name, None)

        stable.sort(key=lambda d: -d.confidence)
        return stable

    def reset(self) -> None:
        """Forget all streak state. Useful for tests and on activation toggles."""
        self._streaks.clear()
