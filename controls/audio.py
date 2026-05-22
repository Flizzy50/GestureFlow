"""System volume control via the Windows Core Audio API (pycaw).

Interaction model is the 'virtual slider grab':
  - On the rising edge of a pinch, capture the current system volume
    and the current pinch ratio as a baseline pair.
  - While pinched, target_volume = base_volume + sensitivity * (base_ratio
    - current_ratio). Tighter pinch than baseline -> louder, looser ->
    quieter, bidirectional from wherever the user grabbed.
  - On release, the most recent applied volume is what stays.

This is direct manipulation: the user is dragging an invisible slider
they grabbed by pinching. Compare with the naive 'absolute mapping' where
the volume jumps to wherever the pinch ratio implies the instant you
pinch — jarring and disconnected from where the system was.
"""
from __future__ import annotations

from typing import Callable, Optional

from controls.base import ActionHandler, RateLimiter
from gestures.base import Detection
from utils.logger import get_logger

log = get_logger(__name__)


# ---------- pycaw bridge ----------
# Module-level cache: creating the IAudioEndpointVolume is expensive and
# the COM object is safe to reuse across calls. First touch initializes.

_endpoint = None  # cached IAudioEndpointVolume instance


def _get_endpoint():
    global _endpoint
    if _endpoint is not None:
        return _endpoint
    # Lazy import so test code (and non-Windows installs) can import this
    # module without pulling in pycaw/comtypes.
    from pycaw.pycaw import AudioUtilities

    # Modern pycaw (post-20220416) wraps IMMDevice in an AudioDevice
    # proxy whose .EndpointVolume property hands back the already-activated
    # IAudioEndpointVolume interface. Older API required us to manually
    # IMMDevice.Activate(IID, ...) + cast through comtypes — that path no
    # longer exists on AudioDevice in current pycaw.
    _endpoint = AudioUtilities.GetSpeakers().EndpointVolume
    return _endpoint


def _system_get_volume() -> float:
    """Current master volume in [0.0, 1.0]."""
    return float(_get_endpoint().GetMasterVolumeLevelScalar())


def _system_set_volume(value: float) -> None:
    """Set master volume. `value` is clamped to [0.0, 1.0] by Core Audio."""
    _get_endpoint().SetMasterVolumeLevelScalar(value, None)


# ---------- handler ----------


class VolumeAction(ActionHandler):
    """Pinch tightness -> system volume, using the virtual-slider model.

    Constructor knobs:
      sensitivity: how much a 1.0-unit change in pinch_ratio shifts volume.
        Typical pinch ratio spans ~0.10 (tight) to ~0.40 (loose); with
        sensitivity=2.5 that full range covers ~75% of the volume range,
        which feels brisk without being twitchy.
      rate_hz: max volume writes per second while pinched.
      get_volume / set_volume: injectable for tests; default to pycaw.
    """

    name = "volume"

    def __init__(
        self,
        sensitivity: float = 2.5,
        rate_hz: float = 12.0,
        get_volume: Optional[Callable[[], float]] = None,
        set_volume: Optional[Callable[[float], None]] = None,
    ) -> None:
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")
        self._sensitivity = sensitivity
        self._rate = RateLimiter(rate_hz)
        self._get_vol = get_volume or _system_get_volume
        self._set_vol = set_volume or _system_set_volume

        self._engaged: bool = False
        self._base_ratio: Optional[float] = None
        self._base_volume: Optional[float] = None

    def update(self, detection: Optional[Detection], now: float) -> bool:
        is_active = detection is not None

        # State transitions: capture baseline on rising edge, drop on release.
        if is_active and not self._engaged:
            ratio = self._extract_ratio(detection)
            if ratio is None:
                # No ratio metadata -> we can't engage. Treat as no detection.
                return False
            self._engaged = True
            self._base_ratio = ratio
            self._base_volume = self._get_vol()
            self._rate.reset()
        elif not is_active and self._engaged:
            self._engaged = False
            self._base_ratio = None
            self._base_volume = None
            return False

        if not self._engaged:
            return False

        if not self._rate.should_fire(now):
            return False

        current_ratio = self._extract_ratio(detection)
        if current_ratio is None:
            return False

        # Gap larger than baseline -> louder. Smaller -> quieter.
        # Sign convention matches the new PinchDetector semantics: thumb
        # and index spread out is "increase", brought closer is "decrease".
        delta = (current_ratio - self._base_ratio) * self._sensitivity
        target = self._base_volume + delta
        if target < 0.0:
            target = 0.0
        elif target > 1.0:
            target = 1.0
        self._set_vol(target)
        return True

    @staticmethod
    def _extract_ratio(detection: Optional[Detection]) -> Optional[float]:
        if detection is None:
            return None
        ratio = detection.metadata.get("pinch_ratio")
        if ratio is None:
            log.warning(
                "VolumeAction received pinch detection without pinch_ratio metadata"
            )
            return None
        return float(ratio)
