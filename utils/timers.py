"""Timing utilities — FPS counters, stopwatches.

Kept here (not in vision/) because anything in the pipeline can time itself:
camera capture, inference, gesture classification, action dispatch.
"""
from __future__ import annotations

import time


class FpsCounter:
    """Exponential-moving-average FPS estimator.

    Why EMA and not a sliding window?
      - O(1) memory and update, no deque bookkeeping.
      - Smoother than instantaneous 1/dt, which jitters every frame.
      - alpha controls responsiveness: higher alpha tracks sudden FPS
        drops faster but is noisier. 0.1 is a reasonable default for a
        ~30 FPS pipeline (effective window ≈ 10 frames).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._ema_dt: float | None = None
        self._last_tick: float | None = None

    def tick(self) -> None:
        """Call once per processed frame."""
        now = time.perf_counter()
        if self._last_tick is not None:
            dt = now - self._last_tick
            self._ema_dt = (
                dt if self._ema_dt is None
                else self._alpha * dt + (1.0 - self._alpha) * self._ema_dt
            )
        self._last_tick = now

    @property
    def fps(self) -> float:
        if not self._ema_dt:
            return 0.0
        return 1.0 / self._ema_dt

    def reset(self) -> None:
        self._ema_dt = None
        self._last_tick = None


class Stopwatch:
    """Context-manager stopwatch for ad-hoc latency measurements.

        with Stopwatch() as sw:
            ...
        log.debug("step took %.2f ms", sw.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
