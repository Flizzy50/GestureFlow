"""Base types for the action layer.

ActionHandler is the seam between gesture detection and OS-level effects.
Every handler:
  - is called once per frame with either its bound gesture's detection
    or None (if the gesture isn't present this frame).
  - owns its own debounce policy via composition (CooldownGate,
    RateLimiter coming in Phase 3.2, etc.).
  - returns whether it fired this call so the dispatcher can log /
    overlay it.

This shape is what lets Phase 4's state machine layer on top without
changing any of this code. The state machine will sit between the
recognizer and the dispatcher, gating which detections are 'real' enough
to forward — but the handler interface stays identical.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional  # noqa: F401  (used by helpers below)

from gestures.base import Detection


@dataclass(frozen=True)
class FiredAction:
    """One action firing — for logging and HUD display."""
    action_name: str
    gesture_name: str
    confidence: float
    fired_at: float  # time.monotonic() value


class ActionHandler(ABC):
    """Per-frame handler with internal debounce/throttle state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable string id used in config bindings and HUD labels."""

    @abstractmethod
    def update(self, detection: Optional[Detection], now: float) -> bool:
        """Tick the handler for one frame.

        Args:
            detection: the current detection for this handler's bound
                       gesture, or None if the gesture isn't detected
                       this frame. Passing None (not skipping the call)
                       is REQUIRED — handlers with edge-triggered
                       semantics need to observe the falling edge.
            now: monotonic timestamp for this frame.

        Returns:
            True if the action executed this call.
        """


class CooldownGate:
    """Rising-edge trigger with a refractory period.

    Lifecycle:
      - gesture absent -> active: candidate to fire.
      - if cooldown has elapsed since last fire, fire and stamp now.
      - subsequent active frames within the cooldown window: do not fire.
      - gesture becomes inactive: arm for the next rising edge.

    This is the right policy for one-shot actions like play/pause where
    holding the gesture should produce exactly one trigger, not 30/sec.
    Volume control (Phase 3.2) uses a different policy (rate limiter).
    """

    def __init__(self, cooldown_seconds: float) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        self._cooldown = cooldown_seconds
        self._last_fire: Optional[float] = None
        self._was_active = False

    def should_fire(self, is_active: bool, now: float) -> bool:
        rising_edge = is_active and not self._was_active
        self._was_active = is_active
        if not rising_edge:
            return False
        if self._last_fire is not None and (now - self._last_fire) < self._cooldown:
            return False
        self._last_fire = now
        return True

    def reset(self) -> None:
        """Forget all state. Useful for tests and on activation toggles."""
        self._last_fire = None
        self._was_active = False


class RateLimiter:
    """Allow a caller to fire at most `max_hz` times per second.

    Counterpart to CooldownGate but for continuous-while-held actions:
      - CooldownGate: 'fire ONCE per gesture press, then quiet'.
      - RateLimiter:  'fire as often as the caller asks, but cap the
                       throughput'.

    Used by VolumeAction (Phase 3.2) and ScrollAction (Phase 3.3) so we
    don't slam pycaw / pyautogui at the full camera frame rate. 10 Hz is
    plenty for both — finer than human ear/eye resolution for the changes
    those actions produce.
    """

    def __init__(self, max_hz: float) -> None:
        if max_hz <= 0:
            raise ValueError("max_hz must be positive")
        self._period = 1.0 / max_hz
        self._last: Optional[float] = None

    def should_fire(self, now: float) -> bool:
        if self._last is None or (now - self._last) >= self._period:
            self._last = now
            return True
        return False

    def reset(self) -> None:
        self._last = None
