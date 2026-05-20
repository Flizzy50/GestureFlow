"""GestureFlow entry point — Phase 1.

Pipeline (Phase 1 scope):
    Camera (thread) -> mirror -> HandTracker -> minimal overlay -> screen
Phase 2 will insert gesture detection between tracker and overlay.
"""
from __future__ import annotations

import sys
from typing import List, Optional

import cv2
import numpy as np

from config import DEFAULT_CONFIG, Config
from gestures.base import Detection
from gestures.recognizer import GestureRecognizer
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


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    n_hands: int,
    top_detection: Optional[Detection],
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


# ---------- pipeline ----------

def run(cfg: Config = DEFAULT_CONFIG) -> int:
    configure_logging(cfg.log_level)
    log.info("GestureFlow phase 1 — press 'q' to quit")

    recognizer = GestureRecognizer([
        OpenPalmDetector(),
        FistDetector(),
        PinchDetector(),
        TwoFingersDetector(),
    ])
    log.info("recognizer loaded with detectors: %s", recognizer.detector_names)

    try:
        with Camera(cfg.camera) as camera, HandTracker(cfg.hand_tracker) as tracker:
            fps = FpsCounter(alpha=0.1)
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

                # Recognize gestures per hand; HUD shows the strongest
                # match across all hands this frame.
                top_detection: Optional[Detection] = None
                for hand in hands:
                    dets = recognizer.process(hand)
                    if dets and (top_detection is None or dets[0].confidence > top_detection.confidence):
                        top_detection = dets[0]

                _draw_hands(frame, hands)
                _draw_hud(frame, fps.fps, len(hands), top_detection)

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
