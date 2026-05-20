"""Routes per-frame detections to bound action handlers."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from controls.base import ActionHandler, FiredAction
from gestures.base import Detection


class ActionDispatcher:
    """Owns gesture-name -> ActionHandler bindings.

    Every frame: collect the best detection per gesture (handles the
    multi-hand case where both hands make the same pose), then tick
    EVERY bound handler with either that detection or None.

    Why every handler ticks every frame: handlers with edge-triggered
    debounce need to observe the falling edge (gesture becoming absent)
    to re-arm for the next rising edge. If we skipped handlers whose
    gesture wasn't detected, they'd never reset.
    """

    def __init__(self, bindings: Mapping[str, ActionHandler]) -> None:
        self._bindings: Dict[str, ActionHandler] = dict(bindings)

    @property
    def bindings(self) -> Mapping[str, ActionHandler]:
        return dict(self._bindings)

    def dispatch(
        self, detections: Sequence[Detection], now: float,
    ) -> List[FiredAction]:
        # Best detection per gesture name. If two hands both make a fist,
        # we collapse to the higher-confidence one — the OS action only
        # needs to fire once.
        best: Dict[str, Detection] = {}
        for det in detections:
            existing = best.get(det.name)
            if existing is None or det.confidence > existing.confidence:
                best[det.name] = det

        fired: List[FiredAction] = []
        for gesture_name, handler in self._bindings.items():
            det = best.get(gesture_name)
            if handler.update(det, now):
                fired.append(
                    FiredAction(
                        action_name=handler.name,
                        gesture_name=gesture_name,
                        confidence=det.confidence if det is not None else 0.0,
                        fired_at=now,
                    )
                )
        return fired
