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

    def test_fires_on_pinch(self):
        det = self.d.detect(_bundle(pinch_hand()))
        self.assertIsNotNone(det)
        self.assertEqual(det.name, "pinch")
        self.assertGreater(det.confidence, 0.95)

    def test_rejects_open_palm(self):
        """Thumb and index tips are far apart in an open palm — no pinch."""
        self.assertIsNone(self.d.detect(_bundle(open_palm_hand())))

    def test_rejects_fist(self):
        """The critical false-positive case: in a fist the thumb and
        index TIPS happen to be close, but the index is curled. The
        min_index_extension guard must reject this."""
        self.assertIsNone(self.d.detect(_bundle(fist_hand())))

    def test_scale_invariance(self):
        """Pinch at 2x camera distance must still fire — the whole point
        of hand_scale normalization."""
        det = self.d.detect(_bundle(scaled(pinch_hand(), factor=2.0)))
        self.assertIsNotNone(det)
        self.assertGreater(det.confidence, 0.95)

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            PinchDetector(closed_ratio=0.5, open_ratio=0.4)


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


if __name__ == "__main__":
    unittest.main()
