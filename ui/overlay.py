"""HUD overlay rendering and linger-state management.

Owns all per-frame drawing — hand skeletons, FPS, gesture labels, and the
'recently fired' banner. Centralized here so:
  - main.py stays focused on pipeline wiring, not display.
  - Linger logic (dynamic gestures and fired actions stay visible for
    longer than they're actually detected) lives next to the rendering
    it controls, in one cohesive object.
  - The visibility predicates can be unit-tested without touching cv2.

Linger semantics:
  - Static gestures (open_palm, fist, ...) show only while currently
    stable — they're persistent state, no linger needed.
  - Dynamic gestures (swipes) are transient: detector fires for ~300ms
    after each motion, then disappears. Without a linger the HUD update
    is gone before the user can register it. We cache and re-show.
  - Fired actions (play_pause, browser_back) flash a banner for the
    same reason — provide visible confirmation that the OS event went
    out, since the OS action itself isn't visible inside our window.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from controls.base import FiredAction
from gestures.base import Detection
from gestures.features import LM, finger_extensions
from vision.hand_tracker import HAND_CONNECTIONS, Hand


_HUD_COLOR = (0, 255, 0)         # green: persistent state (FPS, static gesture)
_DYNAMIC_COLOR = (0, 200, 255)   # amber: transient events (swipes)
_LANDMARK_COLOR = (0, 200, 255)
_CONNECTION_COLOR = (200, 200, 200)
_FIRED_COLOR = (255, 255, 0)     # cyan: action-fired banner

_BAR_LABELS = ("T", "I", "M", "R", "P")  # thumb, index, middle, ring, pinky
_BAR_ON_COLOR = (0, 255, 0)
_BAR_OFF_COLOR = (60, 130, 130)
_BAR_BG_COLOR = (40, 40, 40)
_BAR_THRESHOLD = 0.5             # extension above which a finger reads "on"


class Overlay:
    """Per-frame HUD renderer with linger state for transient events."""

    def __init__(
        self,
        dynamic_linger_seconds: float = 1.0,
        fired_banner_seconds: float = 1.5,
    ) -> None:
        if dynamic_linger_seconds < 0:
            raise ValueError("dynamic_linger_seconds must be non-negative")
        if fired_banner_seconds < 0:
            raise ValueError("fired_banner_seconds must be non-negative")
        self._dynamic_linger = dynamic_linger_seconds
        self._fired_banner = fired_banner_seconds
        self._last_dynamic: Optional[Detection] = None
        self._last_dynamic_at: float = 0.0
        self._last_fired: Optional[FiredAction] = None

    # ----- state input -----

    def note_dynamic(self, detection: Detection, now: float) -> None:
        """Record that a dynamic gesture was detected this frame."""
        self._last_dynamic = detection
        self._last_dynamic_at = now

    def note_fired(self, action: FiredAction) -> None:
        """Record that an action just fired. fired_at lives on the FiredAction."""
        self._last_fired = action

    # ----- visibility (pure, testable) -----

    def dynamic_visible(self, now: float) -> bool:
        return (
            self._last_dynamic is not None
            and (now - self._last_dynamic_at) < self._dynamic_linger
        )

    def fired_visible(self, now: float) -> bool:
        return (
            self._last_fired is not None
            and (now - self._last_fired.fired_at) < self._fired_banner
        )

    # ----- rendering -----

    def draw(
        self,
        frame: np.ndarray,
        fps: float,
        hands: Sequence[Hand],
        top_static: Optional[Detection],
        now: float,
    ) -> None:
        """Render landmarks, HUD, finger bars, and swipe arrow into `frame`."""
        self._draw_hands(frame, hands)
        primary = hands[0] if hands else None
        if primary is not None:
            self._draw_finger_bars(frame, primary)
            self._draw_swipe_arrow(frame, primary, now)
        self._draw_hud(frame, fps, len(hands), top_static, now)

    def _draw_hands(self, frame: np.ndarray, hands: Sequence[Hand]) -> None:
        h, w = frame.shape[:2]
        for hand in hands:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], _CONNECTION_COLOR, 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, _LANDMARK_COLOR, -1, cv2.LINE_AA)

    def _draw_finger_bars(self, frame: np.ndarray, hand: Hand) -> None:
        """Five small vertical bars in the bottom-right showing per-finger
        extension scores. Lets the user see WHY a gesture is or isn't
        firing (e.g., 'fist not registering because thumb still reads 0.6').
        """
        fingers = finger_extensions(hand)
        scores = fingers.as_tuple

        h, w = frame.shape[:2]
        bar_w, bar_h, gap, margin = 14, 64, 8, 20
        x0 = w - margin - (5 * bar_w + 4 * gap)
        y_top = h - margin - bar_h - 18  # leave 18 px below for labels

        for i, (score, label) in enumerate(zip(scores, _BAR_LABELS)):
            x = x0 + i * (bar_w + gap)
            cv2.rectangle(
                frame, (x, y_top), (x + bar_w, y_top + bar_h),
                _BAR_BG_COLOR, -1,
            )
            filled = int(round(bar_h * max(0.0, min(1.0, score))))
            color = _BAR_ON_COLOR if score >= _BAR_THRESHOLD else _BAR_OFF_COLOR
            if filled > 0:
                cv2.rectangle(
                    frame,
                    (x, y_top + bar_h - filled),
                    (x + bar_w, y_top + bar_h),
                    color, -1,
                )
            cv2.putText(
                frame, label, (x + 2, y_top + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

    def _draw_swipe_arrow(
        self, frame: np.ndarray, hand: Hand, now: float,
    ) -> None:
        """Directional arrow from the wrist when a swipe is fresh.

        Anchored at the wrist (most stable single landmark) and lingers
        with the dynamic-gesture banner so the user sees the direction
        for the full 1s window, not just the brief detection moment.
        """
        if not self.dynamic_visible(now):
            return
        assert self._last_dynamic is not None
        name = self._last_dynamic.name
        if name == "swipe_left":
            direction = -1
        elif name == "swipe_right":
            direction = +1
        else:
            return

        h, w = frame.shape[:2]
        wrist = hand.landmark(LM.WRIST)
        cx = int(wrist.x * w)
        cy = int(wrist.y * h)
        arrow_len = int(w * 0.12)  # ~12% of frame width
        start = (cx, cy)
        end = (cx + direction * arrow_len, cy)
        cv2.arrowedLine(
            frame, start, end, _DYNAMIC_COLOR, 5, cv2.LINE_AA, tipLength=0.35,
        )

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        n_hands: int,
        top_static: Optional[Detection],
        now: float,
    ) -> None:
        lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (f"FPS: {fps:5.1f}", _HUD_COLOR),
            (f"hands: {n_hands}", _HUD_COLOR),
        ]
        if top_static is not None:
            pct = int(round(top_static.confidence * 100))
            lines.append((f"{top_static.name}: {pct}%", _HUD_COLOR))
        if self.dynamic_visible(now):
            assert self._last_dynamic is not None  # implied by dynamic_visible
            pct = int(round(self._last_dynamic.confidence * 100))
            lines.append((f">> {self._last_dynamic.name}: {pct}%", _DYNAMIC_COLOR))

        for i, (text, color) in enumerate(lines):
            cv2.putText(
                frame, text, (10, 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

        if self.fired_visible(now):
            assert self._last_fired is not None
            h = frame.shape[0]
            cv2.putText(
                frame, f">> {self._last_fired.action_name}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _FIRED_COLOR, 2, cv2.LINE_AA,
            )
