"""Unit tests for ui.overlay.Overlay's state and visibility predicates.

We do NOT exercise cv2 rendering here — that's integration territory.
What we DO test is the linger-window arithmetic, because that's the
logic that breaks when someone refactors the timing constants and
nobody notices until "swipes flash and disappear too fast" or "the
banner gets stuck forever".
"""
from __future__ import annotations

import unittest

import numpy as np

from controls.base import FiredAction
from gestures.base import Detection
from tests._factories import open_palm_hand
from ui.overlay import Overlay


def _det(name: str = "swipe_left", confidence: float = 0.7) -> Detection:
    return Detection(name=name, confidence=confidence, metadata={})


def _fired(name: str = "play_pause", fired_at: float = 0.0) -> FiredAction:
    return FiredAction(
        action_name=name,
        gesture_name="fist",
        confidence=0.9,
        fired_at=fired_at,
    )


class TestOverlayConstruction(unittest.TestCase):
    def test_invalid_linger_rejected(self):
        with self.assertRaises(ValueError):
            Overlay(dynamic_linger_seconds=-0.1)
        with self.assertRaises(ValueError):
            Overlay(fired_banner_seconds=-0.5)

    def test_initially_nothing_visible(self):
        o = Overlay()
        self.assertFalse(o.dynamic_visible(now=0.0))
        self.assertFalse(o.fired_visible(now=0.0))


class TestDynamicLinger(unittest.TestCase):
    def test_becomes_visible_after_note(self):
        o = Overlay(dynamic_linger_seconds=1.0)
        o.note_dynamic(_det(), now=10.0)
        self.assertTrue(o.dynamic_visible(now=10.5))

    def test_expires_after_linger_window(self):
        o = Overlay(dynamic_linger_seconds=1.0)
        o.note_dynamic(_det(), now=10.0)
        self.assertFalse(o.dynamic_visible(now=11.01))

    def test_visible_at_exact_window_boundary(self):
        """Strict `<` lets the boundary tick be the LAST visible frame.
        Otherwise the banner would flash off one frame earlier than
        expected at typical frame rates."""
        o = Overlay(dynamic_linger_seconds=1.0)
        o.note_dynamic(_det(), now=0.0)
        self.assertTrue(o.dynamic_visible(now=0.999))
        self.assertFalse(o.dynamic_visible(now=1.0))

    def test_later_note_resets_window(self):
        o = Overlay(dynamic_linger_seconds=1.0)
        o.note_dynamic(_det("swipe_left"), now=0.0)
        # Half a window later, a new swipe arrives — window resets.
        o.note_dynamic(_det("swipe_right"), now=0.6)
        self.assertTrue(o.dynamic_visible(now=1.5))   # 0.9s after latest
        self.assertFalse(o.dynamic_visible(now=1.65)) # 1.05s after latest

    def test_zero_linger_makes_dynamic_never_visible(self):
        """Edge case: linger=0 means even the same-frame check fails
        (strict <). Reasonable behavior — effectively disables the linger."""
        o = Overlay(dynamic_linger_seconds=0.0)
        o.note_dynamic(_det(), now=0.0)
        self.assertFalse(o.dynamic_visible(now=0.0))


class TestFiredBanner(unittest.TestCase):
    def test_visible_after_note(self):
        o = Overlay(fired_banner_seconds=1.5)
        o.note_fired(_fired(fired_at=10.0))
        self.assertTrue(o.fired_visible(now=10.5))

    def test_expires_after_banner_window(self):
        o = Overlay(fired_banner_seconds=1.5)
        o.note_fired(_fired(fired_at=10.0))
        self.assertFalse(o.fired_visible(now=11.51))

    def test_uses_fired_at_from_action_not_note_time(self):
        """The banner expires relative to when the action FIRED, not when
        the overlay was notified. Important: there can be a frame of
        latency between firing and noting, and we want the banner to
        appear for the SAME duration regardless."""
        o = Overlay(fired_banner_seconds=1.0)
        # Action fired at t=10.0 but we only noted it at t=10.5.
        o.note_fired(_fired(fired_at=10.0))
        self.assertTrue(o.fired_visible(now=10.9))   # 0.9s after fired_at
        self.assertFalse(o.fired_visible(now=11.0))  # 1.0s after fired_at


class TestOverlayStateIndependence(unittest.TestCase):
    def test_dynamic_and_fired_lingers_are_independent(self):
        """Notes to one channel must not affect the visibility of the other."""
        o = Overlay(dynamic_linger_seconds=1.0, fired_banner_seconds=0.5)
        o.note_dynamic(_det(), now=0.0)
        o.note_fired(_fired(fired_at=0.0))
        # At t=0.6, the fired banner has expired but the dynamic linger hasn't.
        self.assertTrue(o.dynamic_visible(now=0.6))
        self.assertFalse(o.fired_visible(now=0.6))


class TestOverlayDrawSmoke(unittest.TestCase):
    """End-to-end draw() smoke tests: build a frame, exercise every code
    path, ensure no exceptions. Doesn't verify visual output — that's an
    integration concern — but does catch the kind of bug where one of
    the cv2 calls receives a value of the wrong type."""

    def setUp(self):
        # Standard 1280x720 frame, RGB-style channels.
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_draw_with_no_hands(self):
        o = Overlay()
        o.draw(self.frame, fps=17.0, hands=[], top_static=None, now=0.0)
        # If we get here, nothing crashed.

    def test_draw_with_one_hand_and_static_detection(self):
        o = Overlay()
        o.draw(
            self.frame, fps=17.0, hands=[open_palm_hand()],
            top_static=_det("open_palm", 0.95), now=0.0,
        )

    def test_draw_with_dynamic_gesture_lingering(self):
        """Exercises the dynamic-gesture HUD-line code path during the
        linger window."""
        o = Overlay(dynamic_linger_seconds=1.0)
        o.note_dynamic(_det("swipe_left", 0.7), now=0.0)
        o.draw(
            self.frame, fps=17.0, hands=[open_palm_hand()],
            top_static=_det("open_palm", 0.95), now=0.3,
        )

    def test_draw_with_fired_banner_visible(self):
        o = Overlay()
        o.note_fired(_fired("play_pause", fired_at=0.0))
        o.draw(
            self.frame, fps=17.0, hands=[open_palm_hand()],
            top_static=None, now=0.5,
        )

    def test_draw_with_full_state(self):
        """All overlays active simultaneously — the worst-case render."""
        o = Overlay()
        o.note_dynamic(_det("swipe_right", 0.8), now=0.0)
        o.note_fired(_fired("browser_forward", fired_at=0.0))
        o.draw(
            self.frame, fps=17.0, hands=[open_palm_hand()],
            top_static=_det("open_palm", 0.97), now=0.5,
        )

if __name__ == "__main__":
    unittest.main()
