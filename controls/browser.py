"""Browser navigation via the system Back/Forward keyboard shortcut.

Same shape as PlayPauseAction: one-shot edge-triggered key emit, gated by
a CooldownGate to prevent the strobe-fire problem during the ~400 ms
residual-buffer continuation that follows any swipe detection.

Cross-platform note: on Windows and most Linux browsers, Alt+Left/Right
navigates history. macOS uses Cmd+[ and Cmd+] instead — we'd swap the
default key combos in a platform-detection block when adding Mac support.
"""
from __future__ import annotations

from typing import Callable, Optional

from controls.base import ActionHandler, CooldownGate
from gestures.base import Detection


def _press_alt_left() -> None:
    from pynput.keyboard import Controller, Key
    kb = Controller()
    with kb.pressed(Key.alt):
        kb.press(Key.left)
        kb.release(Key.left)


def _press_alt_right() -> None:
    from pynput.keyboard import Controller, Key
    kb = Controller()
    with kb.pressed(Key.alt):
        kb.press(Key.right)
        kb.release(Key.right)


class BrowserBackAction(ActionHandler):
    """Navigate the focused browser back one entry. swipe_left -> Alt+Left."""

    name = "browser_back"

    def __init__(
        self,
        cooldown_seconds: float = 0.8,
        key_press: Optional[Callable[[], None]] = None,
    ) -> None:
        self._gate = CooldownGate(cooldown_seconds)
        self._press = key_press or _press_alt_left

    def update(self, detection: Optional[Detection], now: float) -> bool:
        if self._gate.should_fire(is_active=detection is not None, now=now):
            self._press()
            return True
        return False


class BrowserForwardAction(ActionHandler):
    """Navigate the focused browser forward one entry. swipe_right -> Alt+Right."""

    name = "browser_forward"

    def __init__(
        self,
        cooldown_seconds: float = 0.8,
        key_press: Optional[Callable[[], None]] = None,
    ) -> None:
        self._gate = CooldownGate(cooldown_seconds)
        self._press = key_press or _press_alt_right

    def update(self, detection: Optional[Detection], now: float) -> bool:
        if self._gate.should_fire(is_active=detection is not None, now=now):
            self._press()
            return True
        return False
