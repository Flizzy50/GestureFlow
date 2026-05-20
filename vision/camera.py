"""Threaded webcam capture.

Why threading here matters for real-time CV:
  cv2.VideoCapture.read() is a blocking I/O call. If capture and inference
  share a thread, frame rate is bottlenecked by the slowest stage and you
  silently process stale frames buffered inside the driver.

  Pattern: producer thread reads as fast as the camera allows and stores
  ONLY the latest frame in a 1-slot buffer. The consumer (main loop) pulls
  the freshest frame on demand. Three properties fall out of this:
    1. The consumer never blocks on the camera.
    2. The camera thread never blocks on the consumer.
    3. There is no growing queue — we always work on fresh data, which is
       what matters for interactive gesture control.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import CameraConfig
from utils.logger import get_logger

log = get_logger(__name__)


_BACKENDS = {
    "MSMF": cv2.CAP_MSMF,
    "DSHOW": cv2.CAP_DSHOW,
    "ANY": cv2.CAP_ANY,
}


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or repeatedly fails to read."""


class Camera:
    """Background webcam reader with a single-slot frame buffer."""

    def __init__(self, config: CameraConfig) -> None:
        self._cfg = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._last_consumed_id: int = -1
        self._stop_event = threading.Event()

    # ---------- lifecycle ----------

    def start(self) -> "Camera":
        backend = _BACKENDS.get(self._cfg.backend.upper())
        if backend is None:
            raise CameraError(
                f"unknown camera backend {self._cfg.backend!r}; "
                f"valid: {sorted(_BACKENDS)}"
            )
        self._cap = cv2.VideoCapture(self._cfg.device_index, backend)
        if not self._cap.isOpened():
            raise CameraError(
                f"could not open camera index {self._cfg.device_index} "
                f"via backend {self._cfg.backend}"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps_target)
        # Drop driver-side queue so read() returns the freshest frame, not
        # a 5-frame-old one. Not all drivers honour this, but it's cheap.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(
            "camera opened: %dx%d (requested %dx%d, target %d fps)",
            actual_w, actual_h, self._cfg.width, self._cfg.height, self._cfg.fps_target,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="CameraThread", daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.info("camera stopped")

    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # ---------- I/O ----------

    def _capture_loop(self) -> None:
        assert self._cap is not None
        consecutive_failures = 0
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    log.error("camera read failed 30 times in a row; giving up")
                    self._stop_event.set()
                    break
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            # Single-slot publish. Rebinding (not mutating) means any
            # consumer still holding the previous ndarray is unaffected.
            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1

    def read_new(self) -> Optional[Tuple[np.ndarray, int]]:
        """Return (frame, frame_id) only if a new frame has arrived since
        the last call; otherwise None. Lets the main loop avoid redundant
        inference on the same frame.
        """
        with self._lock:
            if self._latest_frame is None or self._frame_id == self._last_consumed_id:
                return None
            self._last_consumed_id = self._frame_id
            return self._latest_frame, self._frame_id

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
