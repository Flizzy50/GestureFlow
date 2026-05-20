"""Unit tests for gestures.features.

Strategy: the feature layer is pure functions over Hand. Build hands
with known geometric properties, then assert on the derived values.
We use range assertions (>, <) rather than equality so tests survive
minor calibration tweaks without false failures.
"""
from __future__ import annotations

import math
import unittest

from gestures.features import (
    distance,
    finger_extensions,
    hand_scale,
    pinch_distance,
)
from tests._factories import (
    build_hand,
    curled_finger,
    fist_hand,
    lm,
    open_palm_hand,
    pinch_hand,
    scaled,
    straight_finger,
    two_fingers_hand,
)


class TestDistance(unittest.TestCase):
    def test_known_3_4_5_triangle(self):
        a = lm(0.0, 0.0, 0.0)
        b = lm(0.3, 0.4, 0.0)
        self.assertAlmostEqual(distance(a, b), 0.5, places=6)

    def test_distance_to_self_is_zero(self):
        a = lm(0.7, 0.2, 0.1)
        self.assertEqual(distance(a, a), 0.0)

    def test_distance_is_symmetric(self):
        a = lm(0.1, 0.2, 0.3)
        b = lm(0.4, 0.6, 0.5)
        self.assertAlmostEqual(distance(a, b), distance(b, a), places=12)

    def test_distance_uses_z(self):
        a = lm(0.0, 0.0, 0.0)
        b = lm(0.0, 0.0, 1.0)
        self.assertAlmostEqual(distance(a, b), 1.0, places=6)


class TestHandScale(unittest.TestCase):
    def test_known_geometry(self):
        # Wrist at (0.5, 0.9), middle_MCP at (0.5, 0.75) -> distance 0.15
        hand = open_palm_hand()
        self.assertAlmostEqual(hand_scale(hand), 0.15, places=6)


class TestPinchDistance(unittest.TestCase):
    def test_open_hand_pinch_is_large(self):
        # Open palm: thumb and index tips are far apart.
        self.assertGreater(pinch_distance(open_palm_hand()), 0.5)

    def test_pinch_pose_is_small(self):
        # In the canonical pinch hand the thumb tip and index tip coincide.
        self.assertLess(pinch_distance(pinch_hand()), 0.05)

    def test_scale_invariance(self):
        """The whole point of the normalization: same gesture at 2x zoom
        must return the same ratio. If this ever regresses, pinch detection
        silently breaks at non-default camera distances."""
        original = pinch_distance(open_palm_hand())
        zoomed = pinch_distance(scaled(open_palm_hand(), factor=2.0))
        self.assertAlmostEqual(original, zoomed, places=6)

    def test_degenerate_zero_scale_returns_inf(self):
        """Guard against a divide-by-zero corruption on impossible inputs."""
        degenerate = build_hand(
            wrist=(0.5, 0.5, 0.0),
            # All landmarks collapsed to the wrist; hand_scale = 0.
            thumb=[lm(0.5, 0.5) for _ in range(4)],
            index=[lm(0.5, 0.5) for _ in range(4)],
            middle=[lm(0.5, 0.5) for _ in range(4)],
            ring=[lm(0.5, 0.5) for _ in range(4)],
            pinky=[lm(0.5, 0.5) for _ in range(4)],
        )
        self.assertEqual(pinch_distance(degenerate), math.inf)


class TestFingerExtensions(unittest.TestCase):
    def test_open_palm_scores_high_everywhere(self):
        fe = finger_extensions(open_palm_hand())
        for name, score in zip(
            ("thumb", "index", "middle", "ring", "pinky"), fe.as_tuple,
        ):
            self.assertGreater(score, 0.95, msg=f"{name} extension too low: {score:.3f}")

    def test_fist_non_thumb_scores_low(self):
        fe = finger_extensions(fist_hand())
        for name, score in zip(("index", "middle", "ring", "pinky"),
                                (fe.index, fe.middle, fe.ring, fe.pinky)):
            self.assertLess(score, 0.05, msg=f"{name} extension too high: {score:.3f}")

    def test_two_fingers_pose_splits_cleanly(self):
        fe = finger_extensions(two_fingers_hand(thumb_up=False))
        self.assertGreater(fe.index, 0.95)
        self.assertGreater(fe.middle, 0.95)
        self.assertLess(fe.ring, 0.05)
        self.assertLess(fe.pinky, 0.05)

    def test_count_extended_threshold(self):
        fe = finger_extensions(open_palm_hand())
        self.assertEqual(fe.count_extended(threshold=0.7), 5)
        # Raising the threshold past 1.0 means no finger can clear it.
        self.assertEqual(fe.count_extended(threshold=1.01), 0)

    def test_partial_curl_lands_mid_range(self):
        """Mid-bent finger should score in the 0.3-0.9 band, not snap to 0/1.
        Smoothness is what makes confidence usable for Phase 5 visualization."""
        # Construct a finger bent at PIP at roughly 90 degrees:
        # MCP at (0.5, 0.5), PIP straight up at (0.5, 0.3), TIP off to the
        # right at (0.7, 0.3). v1=(0,0.2), v2=(0.2,0) -> cos=0 -> extension≈0.23.
        bent_index = [lm(0.5, 0.5), lm(0.5, 0.3), lm(0.6, 0.3), lm(0.7, 0.3)]
        hand = build_hand(
            wrist=(0.50, 0.90, 0.0),
            thumb=straight_finger(0.40, 0.85, -0.15, -0.10),
            index=bent_index,
            middle=straight_finger(0.50, 0.75, 0.00, -0.45),
            ring=straight_finger(0.55, 0.75,  0.02, -0.40),
            pinky=straight_finger(0.60, 0.78,  0.10, -0.30),
        )
        fe = finger_extensions(hand)
        self.assertGreater(fe.index, 0.10)
        self.assertLess(fe.index, 0.40)


if __name__ == "__main__":
    unittest.main()
