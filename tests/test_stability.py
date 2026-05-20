"""Unit tests for gestures.state_machine.StabilityFilter.

Hysteresis logic is exactly the kind of state-juggling code where edge
cases hide. Each test pins one property; the suite as a whole defines
what "stable" means.
"""
from __future__ import annotations

import unittest

from gestures.base import Detection
from gestures.state_machine import StabilityFilter


def _det(name: str, confidence: float = 0.9, **metadata) -> Detection:
    return Detection(name=name, confidence=confidence, metadata=dict(metadata))


class TestStabilityFilterConstruction(unittest.TestCase):
    def test_invalid_rising_frames_rejected(self):
        with self.assertRaises(ValueError):
            StabilityFilter(rising_frames=0)
        with self.assertRaises(ValueError):
            StabilityFilter(rising_frames=-1)

    def test_threshold_exposed(self):
        self.assertEqual(StabilityFilter(rising_frames=5).rising_threshold, 5)


class TestStabilityFilterBasic(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        f = StabilityFilter(rising_frames=3)
        self.assertEqual(f.update([]), [])

    def test_single_frame_below_threshold_returns_nothing(self):
        f = StabilityFilter(rising_frames=3)
        self.assertEqual(f.update([_det("fist")]), [])

    def test_streak_reaches_threshold_emits(self):
        f = StabilityFilter(rising_frames=3)
        self.assertEqual(f.update([_det("fist")]), [])
        self.assertEqual(f.update([_det("fist")]), [])
        out = f.update([_det("fist")])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].name, "fist")

    def test_continues_emitting_after_stable(self):
        f = StabilityFilter(rising_frames=2)
        f.update([_det("fist")])
        for _ in range(5):
            out = f.update([_det("fist")])
            self.assertEqual(len(out), 1)

    def test_rising_one_fires_immediately(self):
        """rising_frames=1 means no hysteresis — useful for tuning/debug."""
        f = StabilityFilter(rising_frames=1)
        out = f.update([_det("fist")])
        self.assertEqual(len(out), 1)

    def test_absence_resets_streak(self):
        """A single missed frame must reset the counter — otherwise the
        filter would be useless against jitter."""
        f = StabilityFilter(rising_frames=3)
        f.update([_det("fist")])
        f.update([_det("fist")])
        f.update([])  # gesture momentarily absent
        # Streak is back to zero; the next two frames don't get us to 3.
        self.assertEqual(f.update([_det("fist")]), [])
        self.assertEqual(f.update([_det("fist")]), [])


class TestStabilityFilterMultiGesture(unittest.TestCase):
    def test_independent_streaks_per_gesture(self):
        f = StabilityFilter(rising_frames=2)
        f.update([_det("fist")])                          # fist streak=1
        f.update([_det("fist"), _det("open_palm")])       # fist=2 (emits), palm=1
        out = f.update([_det("fist"), _det("open_palm")])  # both stable
        names = {d.name for d in out}
        self.assertEqual(names, {"fist", "open_palm"})

    def test_one_gesture_disappearing_does_not_reset_others(self):
        f = StabilityFilter(rising_frames=3)
        f.update([_det("fist"), _det("open_palm")])  # both =1
        f.update([_det("fist"), _det("open_palm")])  # both =2
        f.update([_det("fist")])                     # open_palm gone, fist=3 -> stable
        out = f.update([_det("fist")])
        self.assertEqual([d.name for d in out], ["fist"])

    def test_output_sorted_by_confidence_desc(self):
        f = StabilityFilter(rising_frames=1)
        out = f.update([
            _det("open_palm", confidence=0.6),
            _det("fist", confidence=0.95),
            _det("pinch", confidence=0.75),
        ])
        self.assertEqual([d.name for d in out], ["fist", "pinch", "open_palm"])


class TestStabilityFilterMultiHand(unittest.TestCase):
    def test_collapses_to_best_confidence_per_gesture(self):
        """Two hands both make a fist with different confidences. The
        filter should track ONE streak for 'fist' and forward the
        higher-confidence detection."""
        f = StabilityFilter(rising_frames=1)
        out = f.update([
            _det("fist", confidence=0.5),
            _det("fist", confidence=0.95),
        ])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].confidence, 0.95)

    def test_same_gesture_from_either_hand_counts_toward_streak(self):
        """As long as SOME hand makes the gesture each frame, the streak
        grows — we're tracking the gesture name, not a particular hand."""
        f = StabilityFilter(rising_frames=3)
        f.update([_det("fist", confidence=0.6)])  # streak=1
        f.update([_det("fist", confidence=0.9)])  # streak=2 (could be other hand)
        out = f.update([_det("fist", confidence=0.7)])  # streak=3 -> stable
        self.assertEqual(len(out), 1)


class TestStabilityFilterMetadataPassthrough(unittest.TestCase):
    def test_stable_detection_carries_latest_metadata(self):
        """When a gesture becomes stable, the emitted Detection has the
        CURRENT frame's metadata (not a snapshot from when the streak
        started). Critical for VolumeAction / ScrollAction which depend
        on fresh per-frame values."""
        f = StabilityFilter(rising_frames=3)
        f.update([_det("pinch", pinch_ratio=0.40)])  # streak=1
        f.update([_det("pinch", pinch_ratio=0.30)])  # streak=2
        out = f.update([_det("pinch", pinch_ratio=0.18)])  # streak=3 -> stable
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].metadata["pinch_ratio"], 0.18)


class TestStabilityFilterStreaks(unittest.TestCase):
    def test_streaks_property_reflects_state(self):
        f = StabilityFilter(rising_frames=5)
        f.update([_det("fist")])
        f.update([_det("fist"), _det("open_palm")])
        s = f.streaks
        self.assertEqual(s["fist"], 2)
        self.assertEqual(s["open_palm"], 1)

    def test_streaks_returned_copy_not_live_view(self):
        """Callers (HUD) must not be able to corrupt internal state."""
        f = StabilityFilter(rising_frames=3)
        f.update([_det("fist")])
        s = f.streaks
        s["fist"] = 999
        self.assertEqual(f.streaks["fist"], 1)


class TestStabilityFilterReset(unittest.TestCase):
    def test_reset_clears_all_streaks(self):
        f = StabilityFilter(rising_frames=3)
        f.update([_det("fist"), _det("open_palm")])
        f.update([_det("fist"), _det("open_palm")])
        f.reset()
        self.assertEqual(f.streaks, {})
        # After reset, the next two frames don't reach the threshold of 3.
        self.assertEqual(f.update([_det("fist")]), [])
        self.assertEqual(f.update([_det("fist")]), [])


if __name__ == "__main__":
    unittest.main()
