"""Unit tests for the swipe detectors.

We construct FeatureBundle directly with a fabricated MotionSnapshot so
detector logic is tested independently of MotionTracker. Tracker behavior
is covered by test_motion.
"""
from __future__ import annotations

import unittest

from gestures.base import FeatureBundle
from gestures.dynamic_gestures import SwipeLeftDetector, SwipeRightDetector
from gestures.features import finger_extensions, hand_scale
from gestures.motion import MotionSnapshot
from tests._factories import fist_hand, open_palm_hand


def _bundle_with_motion(hand, dx: float, dy: float = 0.0, dt: float = 0.2, samples: int = 6) -> FeatureBundle:
    return FeatureBundle(
        hand=hand,
        fingers=finger_extensions(hand),
        hand_scale=hand_scale(hand),
        motion=MotionSnapshot(dx=dx, dy=dy, dt=dt, sample_count=samples),
    )


class TestSwipeLeftDetector(unittest.TestCase):
    def setUp(self):
        self.d = SwipeLeftDetector()

    def test_no_motion_returns_none(self):
        bundle = FeatureBundle(
            hand=open_palm_hand(),
            fingers=finger_extensions(open_palm_hand()),
            hand_scale=hand_scale(open_palm_hand()),
        )
        self.assertIsNone(self.d.detect(bundle))

    def test_fires_on_fast_leftward_open_hand(self):
        # dx=-0.30 over dt=0.2s -> velocity -1.5 -> well above threshold
        b = _bundle_with_motion(open_palm_hand(), dx=-0.30)
        det = self.d.detect(b)
        self.assertIsNotNone(det)
        self.assertEqual(det.name, "swipe_left")
        self.assertGreater(det.confidence, 0.5)

    def test_rejects_rightward_motion(self):
        b = _bundle_with_motion(open_palm_hand(), dx=+0.30)
        self.assertIsNone(self.d.detect(b))

    def test_rejects_slow_motion(self):
        # dx=-0.30 over dt=2.0s -> velocity -0.15 (below 0.8 threshold)
        b = _bundle_with_motion(open_palm_hand(), dx=-0.30, dt=2.0)
        self.assertIsNone(self.d.detect(b))

    def test_rejects_small_displacement(self):
        # Below min_displacement even with fast velocity.
        b = _bundle_with_motion(open_palm_hand(), dx=-0.05, dt=0.05)
        self.assertIsNone(self.d.detect(b))

    def test_rejects_mostly_vertical_motion(self):
        # Big dx, but bigger dy -> looks more like a scroll setup.
        b = _bundle_with_motion(open_palm_hand(), dx=-0.30, dy=-0.40)
        self.assertIsNone(self.d.detect(b))

    def test_rejects_closed_hand(self):
        """Swipe gate: hand must be reasonably open. A 'fist swipe' is
        someone else's gesture."""
        b = _bundle_with_motion(fist_hand(), dx=-0.30)
        self.assertIsNone(self.d.detect(b))

    def test_rejects_too_few_samples(self):
        b = _bundle_with_motion(open_palm_hand(), dx=-0.30, samples=2)
        self.assertIsNone(self.d.detect(b))

    def test_confidence_caps_at_one(self):
        """Very fast swipes don't exceed confidence 1.0."""
        b = _bundle_with_motion(open_palm_hand(), dx=-0.80, dt=0.1)  # velocity -8.0
        det = self.d.detect(b)
        self.assertIsNotNone(det)
        self.assertEqual(det.confidence, 1.0)

    def test_carries_motion_metadata(self):
        b = _bundle_with_motion(open_palm_hand(), dx=-0.30, dy=0.05, dt=0.2)
        det = self.d.detect(b)
        self.assertIsNotNone(det)
        self.assertAlmostEqual(det.metadata["dx"], -0.30)
        self.assertAlmostEqual(det.metadata["dt"], 0.2)
        self.assertAlmostEqual(det.metadata["velocity_x"], -1.5)


class TestSwipeRightDetector(unittest.TestCase):
    def setUp(self):
        self.d = SwipeRightDetector()

    def test_fires_on_fast_rightward_open_hand(self):
        b = _bundle_with_motion(open_palm_hand(), dx=+0.30)
        det = self.d.detect(b)
        self.assertIsNotNone(det)
        self.assertEqual(det.name, "swipe_right")
        self.assertGreater(det.confidence, 0.5)

    def test_rejects_leftward_motion(self):
        b = _bundle_with_motion(open_palm_hand(), dx=-0.30)
        self.assertIsNone(self.d.detect(b))

    def test_rejects_closed_hand(self):
        b = _bundle_with_motion(fist_hand(), dx=+0.30)
        self.assertIsNone(self.d.detect(b))


class TestSwipeMirrorSymmetry(unittest.TestCase):
    """Left and right swipes must be perfect mirror images of each other —
    if asymmetry creeps in, one direction will fire harder than the other."""

    def test_symmetric_confidence_for_symmetric_motion(self):
        left = SwipeLeftDetector()
        right = SwipeRightDetector()
        b_left = _bundle_with_motion(open_palm_hand(), dx=-0.25)
        b_right = _bundle_with_motion(open_palm_hand(), dx=+0.25)
        det_l = left.detect(b_left)
        det_r = right.detect(b_right)
        self.assertIsNotNone(det_l)
        self.assertIsNotNone(det_r)
        self.assertAlmostEqual(det_l.confidence, det_r.confidence, places=6)


class TestSwipeDetectorConstruction(unittest.TestCase):
    def test_invalid_displacement_rejected(self):
        with self.assertRaises(ValueError):
            SwipeLeftDetector(min_displacement=0)

    def test_invalid_velocity_rejected(self):
        with self.assertRaises(ValueError):
            SwipeLeftDetector(min_velocity=-1)


if __name__ == "__main__":
    unittest.main()
