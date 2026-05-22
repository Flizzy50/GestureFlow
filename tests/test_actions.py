"""Unit tests for the action layer.

Three things to verify:
  - CooldownGate's edge-trigger + refractory semantics
  - PlayPauseAction wires the gate to a key-press correctly (with the
    key-press injected as a fake — we never actually press the media key)
  - ActionDispatcher routes detections, picks best-per-gesture across
    hands, and ticks unmatched handlers so they can re-arm
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List, Optional

from controls.audio import VolumeAction
from controls.base import ActionHandler, CooldownGate, FiredAction, RateLimiter
from controls.dispatcher import ActionDispatcher
from controls.media import PlayPauseAction, SkipBackwardAction, SkipForwardAction
from controls.scroll import ScrollAction
from gestures.base import Detection


class TestCooldownGate(unittest.TestCase):
    def test_invalid_cooldown_rejected(self):
        with self.assertRaises(ValueError):
            CooldownGate(cooldown_seconds=-0.5)

    def test_fires_once_on_rising_edge(self):
        g = CooldownGate(cooldown_seconds=1.0)
        self.assertTrue(g.should_fire(is_active=True, now=0.0))

    def test_holds_fire_during_cooldown(self):
        g = CooldownGate(cooldown_seconds=1.0)
        g.should_fire(is_active=True, now=0.0)
        # Stay active; gate must refuse to refire within cooldown.
        for t in (0.1, 0.5, 0.9):
            self.assertFalse(g.should_fire(is_active=True, now=t))

    def test_does_not_fire_while_inactive(self):
        g = CooldownGate(cooldown_seconds=1.0)
        for t in (0.0, 0.1, 0.2):
            self.assertFalse(g.should_fire(is_active=False, now=t))

    def test_refires_after_cooldown_and_new_rising_edge(self):
        g = CooldownGate(cooldown_seconds=0.5)
        self.assertTrue(g.should_fire(is_active=True, now=0.0))
        # Gesture goes away (falling edge re-arms the gate)
        self.assertFalse(g.should_fire(is_active=False, now=0.3))
        # Cooldown elapsed, gesture returns -> rising edge -> fires
        self.assertTrue(g.should_fire(is_active=True, now=0.8))

    def test_falling_edge_alone_does_not_re_enable_within_cooldown(self):
        """Cooldown is enforced even if gesture cycles inside the window."""
        g = CooldownGate(cooldown_seconds=1.0)
        self.assertTrue(g.should_fire(is_active=True, now=0.0))
        # Gesture toggles off/on within cooldown
        self.assertFalse(g.should_fire(is_active=False, now=0.2))
        self.assertFalse(g.should_fire(is_active=True, now=0.3))  # still in cooldown

    def test_reset_clears_state(self):
        g = CooldownGate(cooldown_seconds=1.0)
        g.should_fire(is_active=True, now=0.0)
        g.reset()
        # After reset, the next active call should fire immediately.
        self.assertTrue(g.should_fire(is_active=True, now=0.1))


class TestPlayPauseAction(unittest.TestCase):
    def test_fires_key_on_first_rising_edge(self):
        presses = []
        a = PlayPauseAction(cooldown_seconds=1.0, key_press=lambda: presses.append("press"))
        det = Detection(name="fist", confidence=0.9)
        self.assertTrue(a.update(det, now=0.0))
        self.assertEqual(presses, ["press"])

    def test_does_not_re_fire_while_held(self):
        presses = []
        a = PlayPauseAction(cooldown_seconds=1.0, key_press=lambda: presses.append("press"))
        det = Detection(name="fist", confidence=0.9)
        a.update(det, now=0.0)
        for t in (0.1, 0.5, 0.9):
            a.update(det, now=t)
        self.assertEqual(len(presses), 1)

    def test_re_fires_after_release_and_cooldown(self):
        presses = []
        a = PlayPauseAction(cooldown_seconds=0.5, key_press=lambda: presses.append("press"))
        det = Detection(name="fist", confidence=0.9)
        a.update(det, now=0.0)         # fires
        a.update(None, now=0.3)        # release
        a.update(det, now=0.8)         # rising edge after cooldown -> fires
        self.assertEqual(len(presses), 2)

    def test_handler_name_is_stable(self):
        # ActionHandler.name is used by config bindings; assert it's the
        # expected literal so config strings stay valid.
        self.assertEqual(PlayPauseAction().name, "play_pause")


# A test double that records every update() call.
class _RecordingHandler(ActionHandler):
    name = "recording"

    def __init__(self, fire_on_active: bool = True) -> None:
        self.calls: List[tuple] = []  # (detection_or_None, now)
        self._fire = fire_on_active

    def update(self, detection: Optional[Detection], now: float) -> bool:
        self.calls.append((detection, now))
        return self._fire and detection is not None


class TestActionDispatcher(unittest.TestCase):
    def test_routes_detection_to_bound_handler(self):
        h = _RecordingHandler()
        d = ActionDispatcher({"fist": h})
        fired = d.dispatch([Detection(name="fist", confidence=0.8)], now=1.0)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].action_name, "recording")
        self.assertEqual(fired[0].gesture_name, "fist")
        self.assertAlmostEqual(fired[0].confidence, 0.8)
        self.assertEqual(fired[0].fired_at, 1.0)

    def test_ignores_unbound_gestures(self):
        h = _RecordingHandler()
        d = ActionDispatcher({"fist": h})
        # 'open_palm' has no binding -> handler never ticked for it
        d.dispatch([Detection(name="open_palm", confidence=0.9)], now=1.0)
        self.assertEqual(h.calls, [(None, 1.0)])

    def test_ticks_unmatched_handler_with_none(self):
        """Critical: handlers MUST be ticked every frame so edge-triggered
        gates can observe the falling edge. If we skipped this, CooldownGate
        would never re-arm."""
        h = _RecordingHandler()
        d = ActionDispatcher({"fist": h})
        d.dispatch([], now=2.0)
        self.assertEqual(h.calls, [(None, 2.0)])

    def test_picks_highest_confidence_per_gesture(self):
        """Multi-hand: two hands both make a fist; dispatcher picks the
        higher-confidence one so the action fires exactly once."""
        h = _RecordingHandler()
        d = ActionDispatcher({"fist": h})
        d.dispatch(
            [
                Detection(name="fist", confidence=0.6),
                Detection(name="fist", confidence=0.95),
            ],
            now=1.0,
        )
        # Handler called once, with the higher-conf detection.
        self.assertEqual(len(h.calls), 1)
        det, _ = h.calls[0]
        self.assertIsNotNone(det)
        self.assertAlmostEqual(det.confidence, 0.95)


class TestRateLimiter(unittest.TestCase):
    def test_invalid_hz_rejected(self):
        with self.assertRaises(ValueError):
            RateLimiter(max_hz=0)
        with self.assertRaises(ValueError):
            RateLimiter(max_hz=-5)

    def test_first_call_fires(self):
        r = RateLimiter(max_hz=10)
        self.assertTrue(r.should_fire(now=0.0))

    def test_within_period_does_not_fire(self):
        r = RateLimiter(max_hz=10)  # period = 0.1s
        r.should_fire(now=0.0)
        for t in (0.01, 0.05, 0.09):
            self.assertFalse(r.should_fire(now=t))

    def test_fires_after_period(self):
        r = RateLimiter(max_hz=10)
        r.should_fire(now=0.0)
        self.assertTrue(r.should_fire(now=0.10))

    def test_reset(self):
        r = RateLimiter(max_hz=10)
        r.should_fire(now=0.0)
        r.reset()
        self.assertTrue(r.should_fire(now=0.01))


class _FakeAudio:
    """In-memory audio backend for VolumeAction tests."""
    def __init__(self, initial: float = 0.5) -> None:
        self.volume = initial
        self.writes: List[float] = []

    def get(self) -> float:
        return self.volume

    def set(self, value: float) -> None:
        self.volume = value
        self.writes.append(value)


def _pinch(ratio: float, confidence: float = 0.9) -> Detection:
    return Detection(name="pinch", confidence=confidence, metadata={"pinch_ratio": ratio})


class TestVolumeAction(unittest.TestCase):
    def test_rising_edge_captures_baseline_and_does_not_change_volume(self):
        """The first frame of a pinch must NOT move the volume — it just
        captures where we started. Otherwise the volume jumps the moment
        you pinch, which is exactly what the virtual-slider model avoids."""
        audio = _FakeAudio(initial=0.50)
        v = VolumeAction(sensitivity=2.5, rate_hz=100, get_volume=audio.get, set_volume=audio.set)
        v.update(_pinch(ratio=0.30), now=0.0)
        # First call captures baseline; rate limiter says fire, so set IS called
        # but with delta=0 -> target=base_volume.
        self.assertAlmostEqual(audio.volume, 0.50, places=6)

    def test_larger_gap_raises_volume(self):
        audio = _FakeAudio(initial=0.50)
        v = VolumeAction(sensitivity=2.5, rate_hz=100, get_volume=audio.get, set_volume=audio.set)
        v.update(_pinch(ratio=0.30), now=0.0)   # baseline
        v.update(_pinch(ratio=0.40), now=0.02)  # bigger gap -> louder
        # delta = (0.40 - 0.30) * 2.5 = +0.25
        self.assertAlmostEqual(audio.volume, 0.75, places=6)

    def test_smaller_gap_lowers_volume(self):
        audio = _FakeAudio(initial=0.50)
        v = VolumeAction(sensitivity=2.5, rate_hz=100, get_volume=audio.get, set_volume=audio.set)
        v.update(_pinch(ratio=0.30), now=0.0)
        v.update(_pinch(ratio=0.20), now=0.02)  # smaller gap -> quieter
        # delta = (0.20 - 0.30) * 2.5 = -0.25
        self.assertAlmostEqual(audio.volume, 0.25, places=6)

    def test_volume_clamps_to_unit_interval(self):
        audio = _FakeAudio(initial=0.9)
        v = VolumeAction(sensitivity=10.0, rate_hz=100, get_volume=audio.get, set_volume=audio.set)
        v.update(_pinch(ratio=0.30), now=0.0)
        # Closing the gap hard with huge sensitivity -> wants to go far
        # below 0 -> clamps to 0.0.
        v.update(_pinch(ratio=0.00), now=0.02)
        self.assertEqual(audio.volume, 0.0)
        # Now the other direction.
        v.update(None, now=0.10)                # release
        audio.volume = 0.1
        v.update(_pinch(ratio=0.30), now=0.20)  # new baseline (writes 0.1)
        v.update(_pinch(ratio=0.80), now=0.22)  # opening the gap a lot
        self.assertEqual(audio.volume, 1.0)

    def test_release_then_re_engage_rebases(self):
        """After release, system volume may have been changed by other
        means (the user used the keyboard). The next engagement must use
        the CURRENT volume as the new base, not the one captured before
        release."""
        audio = _FakeAudio(initial=0.40)
        v = VolumeAction(sensitivity=2.0, rate_hz=100, get_volume=audio.get, set_volume=audio.set)
        v.update(_pinch(ratio=0.30), now=0.0)   # baseline 0.40
        v.update(_pinch(ratio=0.40), now=0.02)  # +0.20 -> 0.60
        v.update(None, now=0.10)                # release

        # User cranks volume up via keyboard between gesture sessions
        audio.volume = 0.90

        v.update(_pinch(ratio=0.30), now=0.30)  # new baseline = 0.90
        self.assertAlmostEqual(audio.volume, 0.90, places=6)
        v.update(_pinch(ratio=0.40), now=0.32)  # +0.20 -> 1.0 (clamped from 1.1)
        self.assertEqual(audio.volume, 1.0)

    def test_rate_limit_holds_writes(self):
        audio = _FakeAudio(initial=0.5)
        v = VolumeAction(sensitivity=2.5, rate_hz=10, get_volume=audio.get, set_volume=audio.set)
        v.update(_pinch(ratio=0.30), now=0.0)   # baseline (rate-limited write)
        # Bombard with frames within the rate period — only the first should write.
        n_before = len(audio.writes)
        for i, t in enumerate((0.01, 0.05, 0.09)):
            v.update(_pinch(ratio=0.20 - i * 0.01), now=t)
        self.assertEqual(len(audio.writes), n_before)  # no new writes inside period
        # After the period elapses, the next call writes.
        v.update(_pinch(ratio=0.15), now=0.11)
        self.assertGreater(len(audio.writes), n_before)

    def test_release_returns_false_and_drops_state(self):
        audio = _FakeAudio(initial=0.5)
        v = VolumeAction(sensitivity=2.5, rate_hz=100, get_volume=audio.get, set_volume=audio.set)
        v.update(_pinch(ratio=0.30), now=0.0)
        self.assertFalse(v.update(None, now=0.1))
        # No volume changes recorded after release.
        last_write = audio.writes[-1]
        v.update(None, now=0.2)
        self.assertEqual(audio.writes[-1], last_write)

    def test_missing_pinch_ratio_metadata_is_safe(self):
        """If PinchDetector ever stops emitting pinch_ratio, the handler
        no-ops rather than crashing."""
        audio = _FakeAudio(initial=0.5)
        v = VolumeAction(sensitivity=2.5, rate_hz=100, get_volume=audio.get, set_volume=audio.set)
        bad = Detection(name="pinch", confidence=0.9)  # no metadata
        self.assertFalse(v.update(bad, now=0.0))
        self.assertEqual(audio.writes, [])

    def test_invalid_sensitivity_rejected(self):
        with self.assertRaises(ValueError):
            VolumeAction(sensitivity=0)


def _two_fingers(wrist_y: float, confidence: float = 0.9) -> Detection:
    return Detection(
        name="two_fingers",
        confidence=confidence,
        metadata={"wrist_y": wrist_y},
    )


class _RecordingScroll:
    def __init__(self) -> None:
        self.events: List[int] = []

    def __call__(self, clicks: int) -> None:
        self.events.append(clicks)


class TestScrollAction(unittest.TestCase):
    def test_first_frame_baselines_and_does_not_scroll(self):
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=1, rate_hz=100, scroll_fn=sink)
        self.assertFalse(s.update(_two_fingers(wrist_y=0.5), now=0.0))
        self.assertEqual(sink.events, [])

    def test_hand_up_scrolls_up(self):
        """Hand moving UP means smaller image y. Sign convention: positive
        clicks = scroll up. So this should emit positive."""
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=1, rate_hz=100, scroll_fn=sink)
        s.update(_two_fingers(wrist_y=0.50), now=0.0)   # baseline
        s.update(_two_fingers(wrist_y=0.40), now=0.02)  # hand moved up by 0.10
        # delta = 0.50 - 0.40 = +0.10 -> clicks = round(0.10 * 100) = 10
        self.assertEqual(sink.events, [10])

    def test_hand_down_scrolls_down(self):
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=1, rate_hz=100, scroll_fn=sink)
        s.update(_two_fingers(wrist_y=0.50), now=0.0)
        s.update(_two_fingers(wrist_y=0.60), now=0.02)
        self.assertEqual(sink.events, [-10])

    def test_holding_still_emits_nothing(self):
        """The drag model: stopping motion should stop scrolling. No
        velocity-style continuous scroll while held steady."""
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=1, rate_hz=100, scroll_fn=sink)
        s.update(_two_fingers(wrist_y=0.50), now=0.0)
        for t in (0.02, 0.04, 0.06, 0.08):
            s.update(_two_fingers(wrist_y=0.50), now=t)
        self.assertEqual(sink.events, [])

    def test_re_baseline_after_emit(self):
        """Each emit re-baselines so subsequent ticks measure new motion,
        not cumulative offset. Otherwise holding the hand offset would
        keep scrolling the same total amount each tick."""
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=1, rate_hz=100, scroll_fn=sink)
        s.update(_two_fingers(wrist_y=0.50), now=0.0)
        s.update(_two_fingers(wrist_y=0.40), now=0.02)  # emit 10
        s.update(_two_fingers(wrist_y=0.40), now=0.04)  # still at 0.40 -> no further emit
        self.assertEqual(sink.events, [10])

    def test_jitter_below_min_delta_suppressed(self):
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=3, rate_hz=100, scroll_fn=sink)
        s.update(_two_fingers(wrist_y=0.50), now=0.0)
        # 0.01 motion -> 1 click (< min_delta_clicks=3) -> suppressed
        s.update(_two_fingers(wrist_y=0.49), now=0.02)
        self.assertEqual(sink.events, [])

    def test_jitter_accumulates_until_threshold_crossed(self):
        """Below the jitter floor, baseline must NOT be reset, so steady
        drift eventually crosses the threshold rather than being silently
        eaten frame by frame."""
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=3, rate_hz=100, scroll_fn=sink)
        s.update(_two_fingers(wrist_y=0.50), now=0.0)
        s.update(_two_fingers(wrist_y=0.49), now=0.02)  # 1 click, suppressed
        s.update(_two_fingers(wrist_y=0.48), now=0.04)  # 2 clicks total, still < 3
        s.update(_two_fingers(wrist_y=0.46), now=0.06)  # 4 clicks total, emits
        self.assertEqual(sink.events, [4])

    def test_release_clears_state(self):
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=1, rate_hz=100, scroll_fn=sink)
        s.update(_two_fingers(wrist_y=0.50), now=0.0)
        s.update(None, now=0.1)
        # Re-engage with the SAME y; first frame baselines again -> no scroll.
        s.update(_two_fingers(wrist_y=0.50), now=0.2)
        self.assertEqual(sink.events, [])

    def test_rate_limit_holds_emits(self):
        """Rate limit deliberately does NOT apply to the first post-baseline
        emit — the user wants instant response to motion, not a warmup.
        Subsequent emits within the period are throttled.
        """
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=1, rate_hz=10, scroll_fn=sink)
        s.update(_two_fingers(wrist_y=0.50), now=0.0)   # baseline
        s.update(_two_fingers(wrist_y=0.40), now=0.02)  # first emit — always allowed
        self.assertEqual(len(sink.events), 1)
        s.update(_two_fingers(wrist_y=0.30), now=0.04)  # within 100ms period — held
        self.assertEqual(len(sink.events), 1)
        s.update(_two_fingers(wrist_y=0.20), now=0.15)  # past period — emits
        self.assertEqual(len(sink.events), 2)

    def test_missing_wrist_y_metadata_is_safe(self):
        sink = _RecordingScroll()
        s = ScrollAction(sensitivity=100, min_delta_clicks=1, rate_hz=100, scroll_fn=sink)
        bad = Detection(name="two_fingers", confidence=0.9)
        self.assertFalse(s.update(bad, now=0.0))
        self.assertEqual(sink.events, [])

    def test_invalid_sensitivity_rejected(self):
        with self.assertRaises(ValueError):
            ScrollAction(sensitivity=0)

    def test_invalid_min_delta_rejected(self):
        with self.assertRaises(ValueError):
            ScrollAction(min_delta_clicks=0)


def _swipe(name: str, confidence: float = 0.8) -> Detection:
    return Detection(name=name, confidence=confidence, metadata={})


class TestSkipForwardAction(unittest.TestCase):
    def test_fires_on_first_rising_edge(self):
        presses = []
        a = SkipForwardAction(cooldown_seconds=0.8, key_press=lambda: presses.append("next"))
        self.assertTrue(a.update(_swipe("swipe_left"), now=0.0))
        self.assertEqual(presses, ["next"])

    def test_does_not_re_fire_during_residual_buffer_continuation(self):
        """The whole reason for the 0.8s cooldown: a single swipe leaves
        samples in the motion buffer for ~400ms, so the detector keeps
        firing for that long. The handler must NOT re-press the key."""
        presses = []
        a = SkipForwardAction(cooldown_seconds=0.8, key_press=lambda: presses.append("next"))
        a.update(_swipe("swipe_left"), now=0.0)
        for t in (0.05, 0.10, 0.20, 0.30, 0.40):
            a.update(_swipe("swipe_left"), now=t)
        self.assertEqual(len(presses), 1)

    def test_re_fires_after_release_and_cooldown(self):
        presses = []
        a = SkipForwardAction(cooldown_seconds=0.5, key_press=lambda: presses.append("next"))
        a.update(_swipe("swipe_left"), now=0.0)   # fire
        a.update(None, now=0.45)                  # release (gesture absent)
        a.update(_swipe("swipe_left"), now=0.60)  # past cooldown -> fire
        self.assertEqual(len(presses), 2)

    def test_name(self):
        self.assertEqual(SkipForwardAction().name, "skip_forward")


class TestSkipBackwardAction(unittest.TestCase):
    def test_fires_on_first_rising_edge(self):
        presses = []
        a = SkipBackwardAction(cooldown_seconds=0.8, key_press=lambda: presses.append("prev"))
        self.assertTrue(a.update(_swipe("swipe_right"), now=0.0))
        self.assertEqual(presses, ["prev"])

    def test_independent_handlers_dont_share_state(self):
        """Forward and backward are separate instances; firing one must
        not affect the cooldown of the other."""
        fwd_presses, back_presses = [], []
        fwd = SkipForwardAction(cooldown_seconds=0.8, key_press=lambda: fwd_presses.append("f"))
        back = SkipBackwardAction(cooldown_seconds=0.8, key_press=lambda: back_presses.append("b"))
        fwd.update(_swipe("swipe_left"), now=0.0)
        # Skip-backward should still fire immediately even though
        # skip-forward just fired.
        back.update(_swipe("swipe_right"), now=0.05)
        self.assertEqual(fwd_presses, ["f"])
        self.assertEqual(back_presses, ["b"])

    def test_name(self):
        self.assertEqual(SkipBackwardAction().name, "skip_backward")


if __name__ == "__main__":
    unittest.main()
