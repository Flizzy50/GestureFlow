"""GestureFlow entry point.

Pipeline:
    Camera (thread) -> mirror -> HandTracker -> GestureRecognizer
                    -> StabilityFilter -> ActionDispatcher -> OS effects
                    -> Overlay.draw -> screen
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

import cv2

from config import DEFAULT_CONFIG, Config
from controls.audio import VolumeAction
from controls.base import ActionHandler
from controls.browser import BrowserBackAction, BrowserForwardAction
from controls.dispatcher import ActionDispatcher
from controls.media import PlayPauseAction
from controls.scroll import ScrollAction
from gestures.base import Detection
from gestures.dynamic_gestures import SwipeLeftDetector, SwipeRightDetector
from gestures.motion import MotionTracker
from gestures.recognizer import GestureRecognizer
from gestures.state_machine import StabilityFilter
from gestures.static_gestures import (
    FistDetector,
    OpenPalmDetector,
    PinchDetector,
    TwoFingersDetector,
)
from ui.overlay import Overlay
from utils.logger import configure_logging, get_logger
from utils.stage_timer import StageTimer
from utils.timers import FpsCounter
from vision.async_hand_tracker import AsyncHandTracker
from vision.camera import Camera, CameraError
from vision.hand_tracker import HandTracker


# Profile log cadence: one timing summary every N seconds.
_PROFILE_LOG_INTERVAL = 5.0


# Registry of available action handlers. Config bindings reference these
# by name. Adding a new action = construct it here + bind in config.
def _build_action_handlers() -> Dict[str, ActionHandler]:
    return {
        "play_pause": PlayPauseAction(),
        "volume": VolumeAction(),
        "scroll": ScrollAction(),
        "browser_back": BrowserBackAction(),
        "browser_forward": BrowserForwardAction(),
    }

log = get_logger("gestureflow.main")


# Static vs dynamic partition is a pipeline decision (which detection
# updates persist on the HUD vs flash transiently). Drawing semantics
# live in ui/overlay.py — this constant stays here because main.py is
# the one routing detections into the right HUD lane.
_DYNAMIC_GESTURE_NAMES = frozenset({"swipe_left", "swipe_right"})


# ---------- pipeline ----------

def run(cfg: Config = DEFAULT_CONFIG) -> int:
    configure_logging(cfg.log_level)
    log.info("GestureFlow — press 'q' to quit")

    motion_tracker = MotionTracker(
        window_seconds=cfg.motion.window_seconds,
        max_samples=cfg.motion.max_samples,
    )
    recognizer = GestureRecognizer(
        detectors=[
            OpenPalmDetector(),
            FistDetector(),
            PinchDetector(),
            TwoFingersDetector(),
            SwipeLeftDetector(),
            SwipeRightDetector(),
        ],
        motion_tracker=motion_tracker,
    )
    log.info("recognizer loaded with detectors: %s", recognizer.detector_names)

    stability = StabilityFilter(rising_frames=cfg.stability.rising_frames)
    log.info(
        "stability filter: %d consecutive frames required to engage",
        stability.rising_threshold,
    )

    handlers = _build_action_handlers()
    bindings = {}
    for gesture_name, action_name in cfg.bindings.items():
        handler = handlers.get(action_name)
        if handler is None:
            log.warning(
                "config binds gesture %r to unknown action %r — ignoring",
                gesture_name, action_name,
            )
            continue
        bindings[gesture_name] = handler
    dispatcher = ActionDispatcher(bindings)
    log.info("dispatcher bindings: %s", {g: h.name for g, h in bindings.items()})

    overlay = Overlay()

    timer = StageTimer()
    last_profile_log = 0.0

    try:
        with Camera(cfg.camera) as camera, \
                HandTracker(cfg.hand_tracker) as tracker, \
                AsyncHandTracker(tracker) as async_tracker:
            fps = FpsCounter(alpha=0.1)
            last_rendered_id = -1
            while True:
                # 1) Pump the camera: submit the freshest available frame
                #    for inference. Non-blocking; replaces any pending frame.
                read = camera.read_new()
                if read is not None:
                    frame, _frame_id = read
                    if cfg.mirror_frame:
                        with timer.time("mirror"):
                            frame = cv2.flip(frame, 1)
                    async_tracker.submit(frame, time.monotonic())

                # 2) Render whatever inference last completed. During
                #    warmup (no result yet) show the bare camera frame.
                result = async_tracker.latest()
                if result is None:
                    if read is not None:
                        cv2.imshow(cfg.window_title, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                # 3) Skip the recognize+render path when inference hasn't
                #    produced a NEW result since last loop. We still call
                #    imshow so the GUI thread stays responsive.
                if result.frame_id == last_rendered_id:
                    cv2.imshow(cfg.window_title, result.frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                last_rendered_id = result.frame_id
                now = result.captured_at
                fps.tick()

                with timer.time("motion+recognize"):
                    for hand in result.hands:
                        motion_tracker.update(hand, now)
                    all_detections: List[Detection] = []
                    for hand in result.hands:
                        all_detections.extend(recognizer.process(hand))

                with timer.time("stability+dispatch"):
                    stable_detections = stability.update(all_detections)
                    top_static: Optional[Detection] = next(
                        (d for d in stable_detections if d.name not in _DYNAMIC_GESTURE_NAMES),
                        None,
                    )
                    top_dynamic: Optional[Detection] = next(
                        (d for d in stable_detections if d.name in _DYNAMIC_GESTURE_NAMES),
                        None,
                    )
                    if top_dynamic is not None:
                        overlay.note_dynamic(top_dynamic, now)
                        log.info(
                            "dynamic gesture stable: %s (conf=%.2f)",
                            top_dynamic.name, top_dynamic.confidence,
                        )

                    fired = dispatcher.dispatch(stable_detections, now)
                    if fired:
                        overlay.note_fired(fired[0])
                        log.info(
                            "action fired: %s (from %s, conf=%.2f)",
                            fired[0].action_name, fired[0].gesture_name, fired[0].confidence,
                        )

                with timer.time("render"):
                    overlay.draw(result.frame, fps.fps, result.hands, top_static, now)

                with timer.time("imshow"):
                    cv2.imshow(cfg.window_title, result.frame)
                with timer.time("waitkey"):
                    key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # Periodic timing summary. Note: "inference" is no longer
                # in the timings — it runs on a worker thread now and
                # doesn't gate main-loop iteration time. The FpsCounter
                # measures effective end-to-end throughput.
                if now - last_profile_log >= _PROFILE_LOG_INTERVAL:
                    log.info(
                        "PROFILE ms | fps=%.1f | %s",
                        fps.fps, timer.format_summary(),
                    )
                    last_profile_log = now
    except CameraError as e:
        log.error("camera error: %s", e)
        return 2
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(run())
