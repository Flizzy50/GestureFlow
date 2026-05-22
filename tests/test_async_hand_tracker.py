"""Unit tests for vision.async_hand_tracker.AsyncHandTracker.

Threaded code is harder to test deterministically — we use a mock
tracker that we can pause/release on command, then assert on the
observable state after coordinated wakeups.
"""
from __future__ import annotations

import threading
import time
import unittest
from typing import List

import numpy as np

from vision.async_hand_tracker import AsyncHandTracker, InferenceResult


class _MockTracker:
    """A stand-in for HandTracker with controllable timing.

    By default process() is fast (no sleep). Tests that need to observe
    'worker busy' state set `process_sleep` to make process() block.
    """

    def __init__(self) -> None:
        self.process_count = 0
        self.process_sleep: float = 0.0
        self._gate = threading.Event()
        self._gate.set()  # by default not gated
        self.last_frame_seen: np.ndarray | None = None

    def process(self, frame: np.ndarray) -> List:
        # Record what we received BEFORE the gate so tests can observe
        # "worker has picked up a frame" while the worker is still
        # blocked inside process().
        self.last_frame_seen = frame
        self._gate.wait()
        if self.process_sleep > 0:
            time.sleep(self.process_sleep)
        self.process_count += 1
        return []  # empty hand list — we don't test the hand decode here

    def hold_processing(self) -> None:
        """Block subsequent process() calls until release_processing()."""
        self._gate.clear()

    def release_processing(self) -> None:
        self._gate.set()


def _wait_for(predicate, timeout: float = 1.0, poll: float = 0.005) -> bool:
    """Poll until predicate is true or timeout. Returns whether it became true."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


def _frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


class TestAsyncHandTrackerBasic(unittest.TestCase):
    def test_initial_latest_is_none(self):
        tracker = _MockTracker()
        with AsyncHandTracker(tracker) as a:
            self.assertIsNone(a.latest())

    def test_submit_produces_result(self):
        tracker = _MockTracker()
        with AsyncHandTracker(tracker) as a:
            a.submit(_frame(), captured_at=0.0)
            self.assertTrue(_wait_for(lambda: a.latest() is not None))
            result = a.latest()
            self.assertIsInstance(result, InferenceResult)
            self.assertEqual(result.captured_at, 0.0)

    def test_result_frame_matches_submitted(self):
        tracker = _MockTracker()
        with AsyncHandTracker(tracker) as a:
            f = _frame()
            a.submit(f, captured_at=1.5)
            self.assertTrue(_wait_for(lambda: a.latest() is not None))
            self.assertIs(a.latest().frame, f)

    def test_frame_ids_increment(self):
        tracker = _MockTracker()
        with AsyncHandTracker(tracker) as a:
            a.submit(_frame(), captured_at=0.0)
            self.assertTrue(_wait_for(lambda: a.latest() is not None))
            id1 = a.latest().frame_id

            a.submit(_frame(), captured_at=0.1)
            self.assertTrue(_wait_for(lambda: a.latest().frame_id != id1))
            self.assertGreater(a.latest().frame_id, id1)


class TestAsyncHandTrackerCoalescing(unittest.TestCase):
    def test_rapid_submits_coalesce_to_latest(self):
        """The key invariant: if we submit 10 frames while the worker
        processes one slow inference, the queued frames collapse to ONE
        (the most recent). Anything else would mean a backlog growing
        without bound."""
        tracker = _MockTracker()
        tracker.hold_processing()  # block the first process() call
        with AsyncHandTracker(tracker) as a:
            f1 = _frame()
            a.submit(f1, captured_at=0.0)

            # Let the worker pick up f1 (it will block inside process()).
            self.assertTrue(_wait_for(
                lambda: tracker.last_frame_seen is f1,
                timeout=1.0,
            ))

            # Now flood with 10 frames. Worker is stuck; queue can only
            # hold 1 — older queued frames get dropped.
            frames = [_frame() for _ in range(10)]
            for f in frames:
                a.submit(f, captured_at=0.0)

            # Release the worker. It finishes f1, then picks up exactly
            # one of the queued frames — the most recent submission.
            tracker.release_processing()

            self.assertTrue(_wait_for(
                lambda: tracker.process_count >= 2,
                timeout=1.0,
            ))
            # The worker should NOT have processed all 11 frames.
            self.assertLess(tracker.process_count, 11)
            # The most recent submission should be what was actually used.
            self.assertIs(tracker.last_frame_seen, frames[-1])


class TestAsyncHandTrackerLifecycle(unittest.TestCase):
    def test_close_terminates_worker(self):
        tracker = _MockTracker()
        a = AsyncHandTracker(tracker)
        a.close()
        # Thread should have exited within the join timeout.
        self.assertFalse(a._thread.is_alive())

    def test_close_during_inference_does_not_hang(self):
        """If the worker is mid-process when close() is called, we still
        need to shut down promptly. The worker finishes the in-flight
        frame, then sees the stop flag."""
        tracker = _MockTracker()
        tracker.process_sleep = 0.05  # 50ms inference

        a = AsyncHandTracker(tracker)
        a.submit(_frame(), captured_at=0.0)
        time.sleep(0.01)  # give worker a chance to start the inference
        start = time.monotonic()
        a.close()
        elapsed = time.monotonic() - start
        # Shutdown should be quick: finish one ~50ms inference + small
        # overhead. The 2.0s join timeout is the safety net.
        self.assertLess(elapsed, 1.0)
        self.assertFalse(a._thread.is_alive())

    def test_inference_exception_does_not_kill_worker(self):
        """A bad frame causing process() to throw must not take down the
        whole inference pipeline. Worker logs and continues."""
        class _BrokenTracker(_MockTracker):
            def process(self, frame):
                raise RuntimeError("simulated bad frame")

        broken = _BrokenTracker()
        with AsyncHandTracker(broken) as a:
            a.submit(_frame(), captured_at=0.0)
            time.sleep(0.1)  # let the exception fly
            # Latest should still be None (no result was successfully produced)
            self.assertIsNone(a.latest())
            # But the worker thread should still be alive.
            self.assertTrue(a._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
