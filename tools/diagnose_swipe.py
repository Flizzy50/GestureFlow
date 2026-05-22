"""Live diagnostic for swipe detection — gate-by-gate verdict.

Same camera + motion pipeline as main.py, but instead of dispatching
actions it prints one line per detected hand per frame showing exactly
which swipe-gate (if any) rejected. The trailing column ('gates') reads:

  L_      SwipeLeftDetector fired
  R_      SwipeRightDetector fired
  -:OPN   open-hand gate failed (too few fingers extended)
  -:DSP   net displacement too small
  -:VEL   velocity too low
  -:VRT   motion mostly vertical, not horizontal
  -:SMP   not enough samples buffered yet
  -:DIR   direction mismatch (e.g., dx negative for SwipeRight)
  -:CNF   below confidence floor

ext col is the count_extended(threshold=0.4) — what the loosened gate
sees. The fX columns are per-finger extension scores (thumb..pinky).

Run for ~20 s and perform clear swipes left and right.
  .venv\\Scripts\\python.exe -m tools.diagnose_swipe
"""
from __future__ import annotations

import sys
import time
from typing import Optional, Tuple

import cv2

from config import DEFAULT_CONFIG
from gestures.base import FeatureBundle
from gestures.dynamic_gestures import SwipeLeftDetector, SwipeRightDetector
from gestures.features import finger_extensions, hand_scale
from gestures.motion import MotionSnapshot, MotionTracker
from utils.logger import configure_logging
from vision.camera import Camera
from vision.hand_tracker import HandTracker


# Mirror the loosened defaults so the diagnostic's verdicts match
# what main.py's detectors actually do.
_OPEN_THRESHOLD = 0.4
_MIN_OPEN = 2
_MIN_SAMPLES = 3
_MIN_DISP = 0.15
_MIN_VEL = 0.4
_MAX_VERT_RATIO = 0.7


def _diagnose_gates(
    sign: int, motion: MotionSnapshot, ext_count: int,
) -> Tuple[bool, str]:
    """Replicate _SwipeDetectorBase.detect logic to identify which gate
    rejects. Returns (fired?, short_label_of_failed_gate_or_'_')."""
    if motion.sample_count < _MIN_SAMPLES:
        return False, "SMP"
    if ext_count < _MIN_OPEN:
        return False, "OPN"
    signed_dx = motion.dx * sign
    if signed_dx < _MIN_DISP:
        return False, "DIR" if signed_dx < 0 else "DSP"
    signed_vx = motion.velocity_x * sign
    if signed_vx < _MIN_VEL:
        return False, "VEL"
    if abs(motion.dy) > _MAX_VERT_RATIO * abs(motion.dx):
        return False, "VRT"
    return True, "_"


def _say(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    configure_logging("WARNING")
    cfg = DEFAULT_CONFIG
    motion = MotionTracker(
        window_seconds=cfg.motion.window_seconds,
        max_samples=cfg.motion.max_samples,
    )
    left = SwipeLeftDetector()
    right = SwipeRightDetector()

    _say(">>> swipe diagnostic starting; will run for 20 seconds")
    _say(">>> opening camera...")

    header = (
        f"{'t':>5} {'hand':>5} {'ext':>3} "
        f"{'fT':>4} {'fI':>4} {'fM':>4} {'fR':>4} {'fP':>4} "
        f"{'smp':>3} {'dx':>6} {'dy':>6} {'dt':>5} {'vx':>6} "
        f"{'gates':>10}"
    )

    peak_vx = 0.0

    with Camera(cfg.camera) as camera, HandTracker(cfg.hand_tracker) as tr:
        _say(">>> camera + hand tracker ready. Perform swipes now.")
        _say(header)
        _say("-" * len(header))

        start = time.monotonic()
        last_heartbeat = start
        frames_processed = 0
        frames_with_hand = 0

        while time.monotonic() - start < 20.0:
            r = camera.read_new()
            if r is None:
                time.sleep(0.003)
                continue
            frame, _ = r
            if cfg.mirror_frame:
                frame = cv2.flip(frame, 1)
            hands = tr.process(frame)
            t_rel = time.monotonic() - start
            frames_processed += 1
            if hands:
                frames_with_hand += 1

            # Heartbeat once per second so the user can tell the script
            # is alive even when no hand is being detected.
            if t_rel - (last_heartbeat - start) >= 1.0:
                _say(
                    f"... {t_rel:5.1f}s elapsed | "
                    f"{frames_processed} frames processed, "
                    f"{frames_with_hand} with hand detected"
                )
                last_heartbeat = time.monotonic()

            for hand in hands:
                motion.update(hand, t_rel)

            for hand in hands:
                key = hand.handedness or "Unknown"
                snap = motion.snapshot(key)
                fingers = finger_extensions(hand)
                ext_count = fingers.count_extended(threshold=_OPEN_THRESHOLD)

                if snap is None:
                    _say(
                        f"{t_rel:5.2f} {key[:5]:>5} {ext_count:>3} "
                        f"{fingers.thumb:>4.2f} {fingers.index:>4.2f} "
                        f"{fingers.middle:>4.2f} {fingers.ring:>4.2f} "
                        f"{fingers.pinky:>4.2f} "
                        f"{'-':>3} {'-':>6} {'-':>6} {'-':>5} {'-':>6} "
                        f"{'-:WARM':>10}"
                    )
                    continue

                bundle = FeatureBundle(
                    hand=hand, fingers=fingers,
                    hand_scale=hand_scale(hand), motion=snap,
                )
                fired_l = left.detect(bundle) is not None
                fired_r = right.detect(bundle) is not None
                if fired_l:
                    verdict = "L_"
                elif fired_r:
                    verdict = "R_"
                else:
                    # Find which gate rejected each direction; pick the
                    # one that's "closer" (higher signed_dx).
                    _, label_l = _diagnose_gates(-1, snap, ext_count)
                    _, label_r = _diagnose_gates(+1, snap, ext_count)
                    label = label_l if snap.dx < 0 else label_r
                    verdict = f"-:{label}"

                if abs(snap.velocity_x) > abs(peak_vx):
                    peak_vx = snap.velocity_x

                _say(
                    f"{t_rel:5.2f} {key[:5]:>5} {ext_count:>3} "
                    f"{fingers.thumb:>4.2f} {fingers.index:>4.2f} "
                    f"{fingers.middle:>4.2f} {fingers.ring:>4.2f} "
                    f"{fingers.pinky:>4.2f} "
                    f"{snap.sample_count:>3} {snap.dx:+6.3f} {snap.dy:+6.3f} "
                    f"{snap.dt:5.3f} {snap.velocity_x:+6.2f} "
                    f"{verdict:>10}"
                )

    _say(
        f"\n>>> done. processed {frames_processed} frames, "
        f"{frames_with_hand} with hand detected. "
        f"peak velocity_x = {peak_vx:+.2f}"
    )
    if frames_with_hand == 0:
        _say(
            ">>> WARNING: MediaPipe never detected a hand. "
            "Check lighting and that your hand is in frame."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
