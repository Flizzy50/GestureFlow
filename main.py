"""GestureFlow entry point.

Pipeline:
    Camera (thread) -> mirror -> HandTracker -> GestureRecognizer
                    -> ActionDispatcher -> OS effects
                    -> overlay -> screen
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import DEFAULT_CONFIG, Config
from controls.audio import VolumeAction
from controls.base import ActionHandler, FiredAction
from controls.dispatcher import ActionDispatcher
from controls.media import PlayPauseAction
from controls.scroll import ScrollAction
from gestures.base import Detection
from gestures.recognizer import GestureRecognizer
from gestures.state_machine import StabilityFilter
from gestures.static_gestures import (
    FistDetector,
    OpenPalmDetector,
    PinchDetector,
    TwoFingersDetector,
)
from utils.logger import configure_logging, get_logger
from utils.timers import FpsCounter
from vision.camera import Camera, CameraError
from vision.hand_tracker import HAND_CONNECTIONS, Hand, HandTracker


# Registry of available action handlers. Config bindings reference these
# by name. Adding a new action = construct it here + bind in config.
# Phase 3.2 and 3.3 will extend this with VolumeAction and ScrollAction.
def _build_action_handlers() -> Dict[str, ActionHandler]:
    return {
        "play_pause": PlayPauseAction(),
        "volume": VolumeAction(),
        "scroll": ScrollAction(),
    }

log = get_logger("gestureflow.main")


# ---------- minimal Phase-1 rendering ----------
# A proper overlay module arrives in Phase 5. For now, just enough to see
# that the pipeline is alive and tracking quality is acceptable.

_HUD_COLOR = (0, 255, 0)
_LANDMARK_COLOR = (0, 200, 255)
_CONNECTION_COLOR = (200, 200, 200)


def _draw_hands(frame: np.ndarray, hands: List[Hand]) -> None:
    h, w = frame.shape[:2]
    for hand in hands:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], _CONNECTION_COLOR, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, _LANDMARK_COLOR, -1, cv2.LINE_AA)


_FIRED_BANNER_SECONDS = 1.5  # how long an "action fired!" banner lingers


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    n_hands: int,
    top_detection: Optional[Detection],
    last_fired: Optional[FiredAction],
    now: float,
) -> None:
    lines = [f"FPS: {fps:5.1f}", f"hands: {n_hands}"]
    if top_detection is not None:
        pct = int(round(top_detection.confidence * 100))
        lines.append(f"{top_detection.name}: {pct}%")
    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (10, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _HUD_COLOR, 2, cv2.LINE_AA,
        )
    # Action-fired banner: bottom-left, fades after _FIRED_BANNER_SECONDS.
    if last_fired is not None and (now - last_fired.fired_at) < _FIRED_BANNER_SECONDS:
        h = frame.shape[0]
        cv2.putText(
            frame, f">> {last_fired.action_name}", (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA,
        )


# ---------- pipeline ----------

def run(cfg: Config = DEFAULT_CONFIG) -> int:
    configure_logging(cfg.log_level)
    log.info("GestureFlow — press 'q' to quit")

    recognizer = GestureRecognizer([
        OpenPalmDetector(),
        FistDetector(),
        PinchDetector(),
        TwoFingersDetector(),
    ])
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

    try:
        with Camera(cfg.camera) as camera, HandTracker(cfg.hand_tracker) as tracker:
            fps = FpsCounter(alpha=0.1)
            last_fired: Optional[FiredAction] = None
            while True:
                read = camera.read_new()
                if read is None:
                    # No new frame yet. waitKey(1) yields to the GUI thread
                    # and lets us catch the quit key without burning CPU.
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                frame, _frame_id = read
                if cfg.mirror_frame:
                    # Mirror BEFORE inference so MediaPipe's "Left"/"Right"
                    # labels match what the user sees on screen.
                    frame = cv2.flip(frame, 1)

                hands = tracker.process(frame)
                fps.tick()

                # Recognize per hand, flatten, then debounce.
                all_detections: List[Detection] = []
                for hand in hands:
                    all_detections.extend(recognizer.process(hand))

                stable_detections = stability.update(all_detections)
                top_detection: Optional[Detection] = (
                    stable_detections[0] if stable_detections else None
                )

                now = time.monotonic()
                fired = dispatcher.dispatch(stable_detections, now)
                if fired:
                    last_fired = fired[0]
                    log.info(
                        "action fired: %s (from %s, conf=%.2f)",
                        last_fired.action_name, last_fired.gesture_name, last_fired.confidence,
                    )

                _draw_hands(frame, hands)
                _draw_hud(frame, fps.fps, len(hands), top_detection, last_fired, now)

                cv2.imshow(cfg.window_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except CameraError as e:
        log.error("camera error: %s", e)
        return 2
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(run())
