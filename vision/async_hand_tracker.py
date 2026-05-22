"""Asynchronous wrapper around HandTracker.

The profile data showed inference at 45 ms dominating a 52 ms frame
budget. The other 7 ms (camera read, mirror, recognize, render, GUI)
can't be optimized into sub-30-FPS performance while inference blocks
the main thread. This module is the fix: inference runs on a worker
thread, main loop iterates at camera rate, results are picked up
async.

Why a thread and not a process: MediaPipe's TFLite inference is C++
that releases the Python GIL, so a Python thread genuinely parallelizes
with the main thread's Python-level work. multiprocessing would add
~15 ms of frame-serialization-and-IPC overhead per cycle for no gain.

Single-slot input buffer is the key design decision. If we queued
every frame the camera produced, inference would fall behind by
several frames within seconds. Instead, submitting a new frame
silently drops any older frame still waiting — we always work on
the freshest input.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from utils.logger import get_logger
from vision.hand_tracker import Hand, HandTracker

log = get_logger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """One inference output, packaged with everything needed to render.

    Bundling the frame WITH its hands means we can draw landmarks on
    the exact pixels they were computed from — no risk of overlays
    misaligning with a fresher camera frame that the worker didn't see.
    """
    frame: np.ndarray
    hands: List[Hand]
    captured_at: float
    frame_id: int


# Sentinel pushed into the input queue on close() so the worker wakes
# from its blocking get and notices the stop flag.
_POISON = object()


class AsyncHandTracker:
    """Run HandTracker on a background thread with replace-on-full input."""

    def __init__(self, tracker: HandTracker) -> None:
        self._tracker = tracker
        # maxsize=1 + replace-on-full coalescing pattern. See submit().
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest: Optional[InferenceResult] = None
        self._stop = threading.Event()
        self._counter = 0
        self._thread = threading.Thread(
            target=self._loop, name="InferenceThread", daemon=True,
        )
        self._thread.start()

    def submit(self, frame: np.ndarray, captured_at: float) -> None:
        """Queue a frame for inference. If a previous frame is still
        waiting, it gets dropped — we always want the freshest input."""
        item = (frame, captured_at)
        try:
            self._inbox.put_nowait(item)
        except queue.Full:
            # Slot already has a pending frame the worker hasn't picked
            # up. Drop it and queue this newer one.
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                pass
            try:
                self._inbox.put_nowait(item)
            except queue.Full:
                # Race: worker picked up in between. Just drop this frame
                # — the next submit will catch up.
                pass

    def latest(self) -> Optional[InferenceResult]:
        """Return the most recent completed result, or None if none yet."""
        with self._result_lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        # Wake the worker if it's blocked on the queue.
        try:
            self._inbox.put_nowait(_POISON)
        except queue.Full:
            # Worker will exit on the next iteration via the timeout path.
            pass
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "AsyncHandTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _POISON:
                break
            frame, captured_at = item
            try:
                hands = self._tracker.process(frame)
            except Exception:
                log.exception("inference error; dropping frame")
                continue
            self._counter += 1
            result = InferenceResult(
                frame=frame,
                hands=list(hands),
                captured_at=captured_at,
                frame_id=self._counter,
            )
            with self._result_lock:
                self._latest = result
