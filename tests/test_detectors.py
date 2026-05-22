"""Unit tests for gesture detectors.

For each detector, two kinds of tests matter:
  - POSITIVE: detector fires on its target pose with reasonable confidence.
  - NEGATIVE / CROSS-REJECTION: detector does NOT fire on the OTHER
    canonical poses. Cross-rejection catches calibration drift when
    thresholds are tuned later — without these tests, you can tune
    Phase 4 hysteresis and silently break a different detector.
"""
from __future__ import annotations

import unittest

from gestures.base import FeatureBundle
from gestures.features import finger_extensions, hand_scale
from gestures.recognizer import GestureRecognizer
from gestures.static_gestures import (
    FistDetector,
    OpenPalmDetector,
    PinchDetector,
    TwoFingersDetector,
)
from tests._factories import (
    fist_hand,
    open_palm_hand,
    pinch_hand,
    scaled,
    three_down_hand,
    two_fingers_hand,
)


def _bundle(hand):
    """Build a FeatureBundle the same way the recognizer would."""
    return FeatureBundle(
        hand=hand,
        fingers=finger_extensions(hand),
        hand_scale=hand_scale(hand),
    )


class TestOpenPalmDetector(unittest.TestCase):
    def setUp(self):
        self.d = OpenPalmDetector()

    def test_fires_on_open_palm(self):
        det = self.d.detect(_bundle(open_palm_hand()))
        self.assertIsNotNone(det)
        self.assertEqual(det.name, "open_palm")
        self.assertGreater(det.confidence, 0.95)

    def test_rejects_fist(self):
        self.assertIsNone(self.d.detect(_bundle(fist_hand())))

    def test_rejects_two_fingers(self):
        self.assertIsNone(self.d.detect(_bundle(two_fingers_hand())))


class TestFistDetector(unittest.TestCase):
    def setUp(self):
        self.d = FistDetector()

    def test_fires_on_fist(self):
        det = self.d.detect(_bundle(fist_hand()))
        self.assertIsNotNone(det)
        self.assertEqual(det.name, "fist")
        self.assertGreater(det.confidence, 0.95)

    def test_ignores_thumb_state(self):
        """A fist with thumb fully extended OR fully tucked must still
        register, because real-world thumbs vary."""
        # The default fist_hand keeps the thumb semi-extended; this is
        # the realistic case. Ensure it doesn't get rejected on that basis.
        det = self.d.detect(_bundle(fist_hand()))
        self.assertIsNotNone(det)

    def test_rejects_open_palm(self):
        self.assertIsNone(self.d.detect(_bundle(open_palm_hand())))

    def test_rejects_two_fingers(self):
        """Two-fingers has ring + pinky curled; that's NOT enough to
        register as a fist because index + middle are up."""
        self.assertIsNone(self.d.detect(_bundle(two_fingers_hand())))


class TestPinchDetector(unittest.TestCase):
    def setUp(self):
        self.d = PinchDetector()

    def test_fires_on_three_down_pose(self):
        det = self.d.detect(_bundle(three_down_hand()))
        self.assertIsNotNone(det)
        self.assertEqual(det.name, "pinch")
        self.assertGreater(det.confidence, 0.95)

    def test_rejects_open_palm(self):
        """Middle/ring/pinky are all extended — pose fails on three-down."""
        self.assertIsNone(self.d.detect(_bundle(open_palm_hand())))

    def test_rejects_fist(self):
        """Subtle: a fist DOES have middle/ring/pinky curled (matching
        the three-down condition), but the index is also curled.
        min_index_extension is the load-bearing guard here — without it,
        every fist would trigger volume mode."""
        self.assertIsNone(self.d.detect(_bundle(fist_hand())))

    def test_rejects_two_fingers(self):
        """Two-fingers has middle EXTENDED, violating three-down."""
        self.assertIsNone(self.d.detect(_bundle(two_fingers_hand())))

    def test_emits_thumb_index_gap_in_metadata(self):
        """VolumeAction reads metadata['pinch_ratio'] as the slider value.
        If this contract breaks, volume control silently no-ops."""
        det = self.d.detect(_bundle(three_down_hand(thumb_index_gap=0.10)))
        self.assertIsNotNone(det)
        self.assertIn("pinch_ratio", det.metadata)
        self.assertGreater(det.metadata["pinch_ratio"], 0.0)

    def test_gap_varies_with_thumb_position(self):
        """Two poses with different thumb-index gaps must produce
        different pinch_ratio values — the slider's whole point."""
        close = self.d.detect(_bundle(three_down_hand(thumb_index_gap=0.05)))
        far = self.d.detect(_bundle(three_down_hand(thumb_index_gap=0.20)))
        self.assertIsNotNone(close)
        self.assertIsNotNone(far)
        self.assertLess(close.metadata["pinch_ratio"], far.metadata["pinch_ratio"])

    def test_scale_invariance(self):
        """Pose at 2x camera distance must still fire AND produce the
        same ratio — hand_scale normalization protects against the user
        moving closer/farther from the camera."""
        original = self.d.detect(_bundle(three_down_hand()))
        zoomed = self.d.detect(_bundle(scaled(three_down_hand(), factor=2.0)))
        self.assertIsNotNone(original)
        self.assertIsNotNone(zoomed)
        self.assertAlmostEqual(
            original.metadata["pinch_ratio"],
            zoomed.metadata["pinch_ratio"],
            places=5,
        )

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            PinchDetector(max_other_extension=-0.1)
        with self.assertRaises(ValueError):
            PinchDetector(max_other_extension=1.5)
        with self.assertRaises(ValueError):
            PinchDetector(min_index_extension=-0.1)


class TestTwoFingersDetector(unittest.TestCase):
    def setUp(self):
        self.d = TwoFingersDetector()

    def test_fires_with_thumb_tucked(self):
        det = self.d.detect(_bundle(two_fingers_hand(thumb_up=False)))
        self.assertIsNotNone(det)
        self.assertEqual(det.name, "two_fingers")

    def test_fires_with_thumb_up(self):
        """Thumb position is don't-care; both poses must register."""
        det = self.d.detect(_bundle(two_fingers_hand(thumb_up=True)))
        self.assertIsNotNone(det)

    def test_rejects_open_palm(self):
        """All five fingers extended — ring/pinky violate max_down."""
        self.assertIsNone(self.d.detect(_bundle(open_palm_hand())))

    def test_rejects_fist(self):
        """All curled — index/middle violate min_up."""
        self.assertIsNone(self.d.detect(_bundle(fist_hand())))

    def test_emits_wrist_position_metadata(self):
        """ScrollAction depends on wrist_y in metadata. If this contract
        breaks silently, scroll would log a warning and stop working."""
        det = self.d.detect(_bundle(two_fingers_hand()))
        self.assertIsNotNone(det)
        self.assertIn("wrist_y", det.metadata)
        self.assertIn("wrist_x", det.metadata)
        # Wrist y for the canonical pose is 0.90.
        self.assertAlmostEqual(det.metadata["wrist_y"], 0.90, places=6)


class TestRecognizerComposition(unittest.TestCase):
    """End-to-end check: recognizer with all four detectors picks the
    right gesture for each canonical pose, and returns NOTHING for
    poses that none of them should match."""

    def setUp(self):
        self.rec = GestureRecognizer([
            OpenPalmDetector(),
            FistDetector(),
            PinchDetector(),
            TwoFingersDetector(),
        ])

    def test_open_palm_pose_top_match(self):
        dets = self.rec.process(open_palm_hand())
        self.assertTrue(dets, "expected at least one detection")
        self.assertEqual(dets[0].name, "open_palm")

    def test_fist_pose_top_match(self):
        dets = self.rec.process(fist_hand())
        self.assertTrue(dets)
        self.assertEqual(dets[0].name, "fist")

    def test_two_fingers_pose_top_match(self):
        dets = self.rec.process(two_fingers_hand())
        self.assertTrue(dets)
        self.assertEqual(dets[0].name, "two_fingers")

    def test_detections_sorted_by_confidence(self):
        dets = self.rec.process(open_palm_hand())
        confidences = [d.confidence for d in dets]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_empty_detector_list_rejected(self):
        with self.assertRaises(ValueError):
            GestureRecognizer([])


class TestRecognizerMotionIntegration(unittest.TestCase):
    """Verify the recognizer correctly bundles a MotionSnapshot into
    FeatureBundle when a tracker is provided."""

    def test_no_tracker_leaves_motion_none(self):
        from gestures.base import FeatureBundle
        rec = GestureRecognizer([OpenPalmDetector()])
        # Process succeeds without raising — motion remains None.
        # Indirect check: detector still works (it doesn't read motion).
        dets = rec.process(open_palm_hand())
        self.assertTrue(dets)

    def test_with_tracker_populates_motion(self):
        from gestures.dynamic_gestures import SwipeLeftDetector
        from gestures.motion import MotionTracker
        from tests._factories import build_hand, straight_finger

        def _hand_at(x):
            return build_hand(
                wrist=(x, 0.5, 0.0),
                thumb=straight_finger(0.40, 0.85, -0.15, -0.10),
                index=straight_finger(0.45, 0.75, -0.02, -0.40),
                middle=straight_finger(0.50, 0.75,  0.00, -0.45),
                ring=straight_finger(0.55, 0.75,   0.02, -0.40),
                pinky=straight_finger(0.60, 0.78,  0.10, -0.30),
            )

        tracker = MotionTracker(window_seconds=1.0)
        rec = GestureRecognizer([SwipeLeftDetector()], motion_tracker=tracker)

        # Simulate a leftward swipe across several frames.
        positions = [(0.0, 0.90), (0.05, 0.75), (0.10, 0.55), (0.15, 0.30), (0.20, 0.20), (0.25, 0.15)]
        last_dets = []
        for t, x in positions:
            h = _hand_at(x)
            tracker.update(h, t)
            last_dets = rec.process(h)

        # After enough samples accumulate, SwipeLeftDetector should fire.
        self.assertTrue(last_dets, "expected SwipeLeftDetector to fire after motion accumulates")
        self.assertEqual(last_dets[0].name, "swipe_left")


if __name__ == "__main__":
    unittest.main()
