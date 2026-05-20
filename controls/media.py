"""Media controls — keys sent to whichever media app currently owns focus
of the system media transport (Spotify, VLC, YouTube in a focused tab, etc.).

We use pynput's virtual-key abstraction. On Windows this maps to
VK_MEDIA_PLAY_PAUSE (0xB3); on Linux/macOS pynput translates to the
platform-appropriate XF86 / NSEvent equivalents. Cross-platform for free.
"""
from __future__ import annotations

from typing import Callable, Optional

from controls.base import ActionHandler, CooldownGate
from gestures.base import Detection


def _press_media_play_pause() -> None:
    """Send the system media play/pause key.

    Lazy pynput import: test code (and CI without an X display) can
    construct PlayPauseAction with an injected key_press callable and
    never touch pynput. Importing inside the function keeps the module
    importable even if pynput isn't installed yet.
    """
    from pynput.keyboard import Controller, Key
    kb = Controller()
    kb.press(Key.media_play_pause)
    kb.release(Key.media_play_pause)


class PlayPauseAction(ActionHandler):
    """Single-shot media play/pause toggle.

    Holding the gesture continuously fires exactly once, then waits
    `cooldown_seconds` before being eligible again. This is critical:
    without the cooldown, a half-second fist (~8 frames at 17 FPS) would
    toggle Spotify play/pause eight times — leaving you back where you
    started or worse, in a strobe.

    The default 0.8s cooldown is long enough to outlast normal hand
    wobble at gesture boundaries; short enough that a deliberate
    "pause-then-play" two-tap is still fluid.
    """

    name = "play_pause"

    def __init__(
        self,
        cooldown_seconds: float = 0.8,
        key_press: Optional[Callable[[], None]] = None,
    ) -> None:
        self._gate = CooldownGate(cooldown_seconds)
        self._press = key_press or _press_media_play_pause

    def update(self, detection: Optional[Detection], now: float) -> bool:
        if self._gate.should_fire(is_active=detection is not None, now=now):
            self._press()
            return True
        return False
