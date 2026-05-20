"""Two-finger gesture -> drag scrolling.

Model: incremental drag.
  - On rising edge of two-fingers gesture, capture wrist_y as the
    baseline reference point. Do NOT emit scroll on the first frame.
  - On subsequent ticks (rate-limited), compute delta_y = baseline - now.
    Hand moved UP in image space (smaller y) -> positive delta -> scroll up.
  - After emitting a scroll, re-baseline to the current y so the next
    tick measures the next frame-to-frame motion, not cumulative offset.
  - On release, drop state.

Why re-baseline after each emit (not just at start):
  We want the page to follow the hand's MOTION, not its absolute offset
  from the start point. A user who scrolls down half a page and then
  holds steady should see scrolling STOP, not continue at the offset.
  Re-baselining each fire achieves "drag" semantics — same as touchpad.

Scroll target window: pynput.mouse.scroll() targets whichever window the
mouse cursor is over. The gesture system doesn't move the cursor; the
user must hover the cursor over the scrollable thing first. This is the
honest constraint of OS-level scroll emission.
"""
from __future__ import annotations

from typing import Callable, Optional

from controls.base import ActionHandler, RateLimiter
from gestures.base import Detection
from utils.logger import get_logger

log = get_logger(__name__)


def _system_scroll(clicks: int) -> None:
    """Emit a vertical scroll event. Positive clicks = scroll up."""
    from pynput.mouse import Controller
    Controller().scroll(0, clicks)


class ScrollAction(ActionHandler):
    """Vertical-drag scrolling tied to wrist_y of the two-fingers gesture.

    Constructor knobs:
      sensitivity: clicks per unit of normalized y-delta. With y in [0, 1],
        moving the hand ~0.05 (a small flick) at sensitivity=60 = 3 clicks,
        roughly one "notch" of a typical scroll wheel. Tune to taste.
      min_delta_clicks: ignore micro-motion smaller than this to suppress
        jitter from MediaPipe landmark noise.
      rate_hz: max scroll emits per second. 15 Hz feels smooth; higher
        risks event flooding the foreground app.
      scroll_fn: injectable for tests.
    """

    name = "scroll"

    def __init__(
        self,
        sensitivity: float = 60.0,
        min_delta_clicks: int = 1,
        rate_hz: float = 15.0,
        scroll_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")
        if min_delta_clicks < 1:
            raise ValueError("min_delta_clicks must be >= 1")
        self._sensitivity = sensitivity
        self._min_delta = min_delta_clicks
        self._rate = RateLimiter(rate_hz)
        self._scroll = scroll_fn or _system_scroll

        self._engaged: bool = False
        self._base_y: Optional[float] = None

    def update(self, detection: Optional[Detection], now: float) -> bool:
        is_active = detection is not None

        if is_active and not self._engaged:
            y = self._extract_y(detection)
            if y is None:
                return False
            self._engaged = True
            self._base_y = y
            self._rate.reset()
            return False  # baseline frame, no scroll
        elif not is_active and self._engaged:
            self._engaged = False
            self._base_y = None
            return False

        if not self._engaged:
            return False

        if not self._rate.should_fire(now):
            return False

        y = self._extract_y(detection)
        if y is None:
            return False

        # Image y grows downward, so moving the hand up DECREASES y.
        # Positive delta -> scroll up (matches user intuition + pynput sign).
        delta_y = self._base_y - y
        clicks = int(round(delta_y * self._sensitivity))
        if abs(clicks) < self._min_delta:
            # Below the jitter floor — don't emit and don't re-baseline,
            # so accumulated micro-motion eventually crosses the threshold.
            return False

        self._scroll(clicks)
        # Re-baseline so the next tick measures the next motion, not the
        # cumulative offset from the original baseline.
        self._base_y = y
        return True

    @staticmethod
    def _extract_y(detection: Optional[Detection]) -> Optional[float]:
        if detection is None:
            return None
        y = detection.metadata.get("wrist_y")
        if y is None:
            log.warning(
                "ScrollAction received two_fingers detection without wrist_y metadata"
            )
            return None
        return float(y)
