"""Unit tests for gestures.motion.MotionTracker."""
from __future__ import annotations

import unittest

from gestures.motion import MotionSample, MotionSnapshot, MotionTracker
from tests._factories import build_hand, straight_finger


def _hand(handedness: str, wrist_x: float = 0.5, wrist_y: float = 0.5):
    """Build a minimal hand at a specific wrist location."""
    return build_hand(
        wrist=(wrist_x, wrist_y, 0.0),
        thumb=straight_finger(0.40, 0.85, -0.15, -0.10),
        index=straight_finger(0.45, 0.75, -0.02, -0.40),
        middle=straight_finger(0.50, 0.75,  0.00, -0.45),
        ring=straight_finger(0.55, 0.75,   0.02, -0.40),
        pinky=straight_finger(0.60, 0.78,  0.10, -0.30),
        handedness=handedness,
    )


class TestMotionTrackerConstruction(unittest.TestCase):
    def test_invalid_window_rejected(self):
        with self.assertRaises(ValueError):
            MotionTracker(window_seconds=0)
        with self.assertRaises(ValueError):
            MotionTracker(window_seconds=-0.1)

    def test_invalid_max_samples_rejected(self):
        with self.assertRaises(ValueError):
            MotionTracker(max_samples=1)


class TestMotionTrackerBasic(unittest.TestCase):
    def test_no_samples_returns_none(self):
        m = MotionTracker()
        self.assertIsNone(m.snapshot("Right"))

    def test_single_sample_returns_none(self):
        """Need at least two samples to compute displacement."""
        m = MotionTracker()
        m.update(_hand("Right", wrist_x=0.5), now=0.0)
        self.assertIsNone(m.snapshot("Right"))

    def test_basic_displacement(self):
        m = MotionTracker(window_seconds=1.0)
        m.update(_hand("Right", wrist_x=0.30), now=0.0)
        m.update(_hand("Right", wrist_x=0.70), now=0.2)
        snap = m.snapshot("Right")
        self.assertIsNotNone(snap)
        self.assertAlmostEqual(snap.dx, 0.40, places=6)
        self.assertAlmostEqual(snap.dt, 0.20, places=6)
        self.assertEqual(snap.sample_count, 2)

    def test_velocity_derived_from_dx_and_dt(self):
        m = MotionTracker(window_seconds=1.0)
        m.update(_hand("Right", wrist_x=0.30), now=0.0)
        m.update(_hand("Right", wrist_x=0.50), now=0.1)
        snap = m.snapshot("Right")
        # dx=0.20, dt=0.1 -> velocity=2.0
        self.assertAlmostEqual(snap.velocity_x, 2.0, places=6)


class TestMotionTrackerWindow(unittest.TestCase):
    def test_samples_outside_window_expire(self):
        m = MotionTracker(window_seconds=0.5)
        m.update(_hand("Right", wrist_x=0.0), now=0.0)
        m.update(_hand("Right", wrist_x=0.3), now=0.2)
        # This update is at t=1.0, which is 1.0s past the first sample (0.0)
        # and 0.8s past the second (0.2). Both fall outside the 0.5s window.
        m.update(_hand("Right", wrist_x=0.9), now=1.0)
        snap = m.snapshot("Right")
        # Only the last sample survives -> not enough for a snapshot.
        self.assertIsNone(snap)

    def test_partial_window_expiry(self):
        m = MotionTracker(window_seconds=0.5)
        m.update(_hand("Right", wrist_x=0.0), now=0.0)
        m.update(_hand("Right", wrist_x=0.3), now=0.3)
        m.update(_hand("Right", wrist_x=0.5), now=0.4)
        # At t=0.6, the first sample (t=0.0) is outside the 0.5s window.
        m.update(_hand("Right", wrist_x=0.7), now=0.6)
        snap = m.snapshot("Right")
        # First should be the one from t=0.3
        self.assertAlmostEqual(snap.dx, 0.4, places=6)  # 0.7 - 0.3
        self.assertEqual(snap.sample_count, 3)


class TestMotionTrackerPerHand(unittest.TestCase):
    def test_independent_buffers_per_handedness(self):
        m = MotionTracker(window_seconds=1.0)
        m.update(_hand("Right", wrist_x=0.2), now=0.0)
        m.update(_hand("Left",  wrist_x=0.8), now=0.0)
        m.update(_hand("Right", wrist_x=0.9), now=0.2)
        m.update(_hand("Left",  wrist_x=0.1), now=0.2)
        right = m.snapshot("Right")
        left = m.snapshot("Left")
        self.assertAlmostEqual(right.dx, +0.7, places=6)
        self.assertAlmostEqual(left.dx, -0.7, places=6)

    def test_unknown_handedness_lands_in_its_own_bucket(self):
        m = MotionTracker(window_seconds=1.0)
        m.update(_hand("", wrist_x=0.3), now=0.0)  # empty -> "Unknown"
        m.update(_hand("", wrist_x=0.6), now=0.1)
        self.assertIsNone(m.snapshot("Right"))
        snap = m.snapshot("Unknown")
        self.assertIsNotNone(snap)
        self.assertAlmostEqual(snap.dx, 0.3, places=6)


class TestMotionTrackerReset(unittest.TestCase):
    def test_reset_clears_all_buffers(self):
        m = MotionTracker(window_seconds=1.0)
        m.update(_hand("Right", wrist_x=0.0), now=0.0)
        m.update(_hand("Right", wrist_x=0.5), now=0.1)
        m.update(_hand("Left",  wrist_x=0.5), now=0.1)
        m.reset()
        self.assertIsNone(m.snapshot("Right"))
        self.assertIsNone(m.snapshot("Left"))


class TestMotionSnapshot(unittest.TestCase):
    def test_velocity_zero_when_dt_zero(self):
        """Defensive: snapshot velocity properties handle dt=0 gracefully."""
        snap = MotionSnapshot(dx=0.5, dy=0.0, dt=0.0, sample_count=2)
        self.assertEqual(snap.velocity_x, 0.0)
        self.assertEqual(snap.velocity_y, 0.0)


if __name__ == "__main__":
    unittest.main()
