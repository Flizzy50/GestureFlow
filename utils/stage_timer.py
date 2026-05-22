"""Per-stage exponential moving-average timing.

Designed to be cheap enough to leave on in production. perf_counter is
sub-microsecond; the EMA update is a few floats. The user-facing payoff
is that you always know where your milliseconds are going — no
'is the pipeline slow today? let me re-add my timing prints' moments.

Usage:

    timer = StageTimer()
    with timer.time("inference"):
        result = expensive_call()

    timer.summary()  # {"inference": 35.2, ...}  (milliseconds, EMA)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


class StageTimer:
    """Tracks an EMA of milliseconds per named stage."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._ema_ms: Dict[str, float] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            prev = self._ema_ms.get(name)
            if prev is None:
                self._ema_ms[name] = dt_ms
            else:
                self._ema_ms[name] = self._alpha * dt_ms + (1.0 - self._alpha) * prev

    def record(self, name: str, dt_ms: float) -> None:
        """Direct injection — for tests, or for stages already timed elsewhere."""
        prev = self._ema_ms.get(name)
        if prev is None:
            self._ema_ms[name] = dt_ms
        else:
            self._ema_ms[name] = self._alpha * dt_ms + (1.0 - self._alpha) * prev

    def summary(self) -> Dict[str, float]:
        """Snapshot of current EMA values, in milliseconds. Returns a
        copy so callers can't mutate internal state."""
        return dict(self._ema_ms)

    def format_summary(self) -> str:
        """One-line human-readable rendering, ordered by descending cost.
        Useful for log lines: ``inference=37.2 render=5.1 ...``"""
        return " ".join(
            f"{name}={ms:.1f}"
            for name, ms in sorted(
                self._ema_ms.items(), key=lambda kv: -kv[1]
            )
        )

    def reset(self) -> None:
        self._ema_ms.clear()
