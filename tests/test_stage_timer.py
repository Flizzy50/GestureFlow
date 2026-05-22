"""Unit tests for utils.stage_timer.StageTimer."""
from __future__ import annotations

import time
import unittest

from utils.stage_timer import StageTimer


class TestStageTimerConstruction(unittest.TestCase):
    def test_invalid_alpha_rejected(self):
        with self.assertRaises(ValueError):
            StageTimer(alpha=0)
        with self.assertRaises(ValueError):
            StageTimer(alpha=-0.5)
        with self.assertRaises(ValueError):
            StageTimer(alpha=1.5)

    def test_alpha_of_one_is_allowed(self):
        # alpha=1.0 means "no smoothing" — each sample replaces the EMA.
        # Useful for tests; the constructor must accept it.
        StageTimer(alpha=1.0)


class TestStageTimerRecord(unittest.TestCase):
    def test_first_record_sets_value_directly(self):
        t = StageTimer(alpha=0.5)
        t.record("foo", 10.0)
        self.assertEqual(t.summary()["foo"], 10.0)

    def test_subsequent_records_compute_ema(self):
        t = StageTimer(alpha=0.5)
        t.record("foo", 10.0)
        t.record("foo", 20.0)
        # EMA: 0.5*20 + 0.5*10 = 15.0
        self.assertAlmostEqual(t.summary()["foo"], 15.0)

    def test_alpha_one_replaces_each_sample(self):
        t = StageTimer(alpha=1.0)
        t.record("foo", 10.0)
        t.record("foo", 30.0)
        self.assertEqual(t.summary()["foo"], 30.0)

    def test_multiple_stages_independent(self):
        t = StageTimer(alpha=0.5)
        t.record("a", 10.0)
        t.record("b", 100.0)
        self.assertEqual(t.summary()["a"], 10.0)
        self.assertEqual(t.summary()["b"], 100.0)


class TestStageTimerContextManager(unittest.TestCase):
    def test_time_context_records_elapsed_in_ms(self):
        """We can't assert exact timing but a 10ms sleep should land in
        the 5-50ms range under any sane scheduler."""
        t = StageTimer(alpha=1.0)
        with t.time("sleep"):
            time.sleep(0.01)
        elapsed = t.summary()["sleep"]
        self.assertGreater(elapsed, 5.0)
        self.assertLess(elapsed, 100.0)

    def test_time_context_records_even_on_exception(self):
        """try/finally semantics: a stage that raises still gets timed.
        Without this, exceptions during inference would silently disable
        the profiler for that stage."""
        t = StageTimer(alpha=1.0)
        try:
            with t.time("bad"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIn("bad", t.summary())


class TestStageTimerSummary(unittest.TestCase):
    def test_summary_is_a_copy(self):
        """Callers must not be able to corrupt internal state by mutating
        the snapshot. Tested because dict() copy semantics are exactly the
        sort of thing that gets refactored to a live reference."""
        t = StageTimer(alpha=1.0)
        t.record("foo", 10.0)
        snap = t.summary()
        snap["foo"] = 999.0
        self.assertEqual(t.summary()["foo"], 10.0)

    def test_format_summary_ordered_by_descending_cost(self):
        """The most expensive stage comes first so it catches the eye in
        a log line."""
        t = StageTimer(alpha=1.0)
        t.record("a", 5.0)
        t.record("b", 50.0)
        t.record("c", 25.0)
        out = t.format_summary()
        # Order should be b, c, a (50, 25, 5)
        self.assertEqual(out, "b=50.0 c=25.0 a=5.0")

    def test_format_summary_empty(self):
        self.assertEqual(StageTimer().format_summary(), "")


class TestStageTimerReset(unittest.TestCase):
    def test_reset_clears_all_stages(self):
        t = StageTimer(alpha=1.0)
        t.record("foo", 10.0)
        t.record("bar", 20.0)
        t.reset()
        self.assertEqual(t.summary(), {})


if __name__ == "__main__":
    unittest.main()
