"""Test helpers for fabricating Hand objects.

Why this module exists: every Hand needs 21 landmarks in a specific
order, and we want tests that read as "build pose, assert property".
The underscore prefix keeps unittest's test discovery from trying to
load this file as a test module.

Coordinate convention follows MediaPipe: x in [0, 1] across image width,
y in [0, 1] down the image height (so y=0.5 is the vertical centerline,
y=0.9 is near the bottom — where a wrist would naturally sit).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from vision.hand_tracker import Hand, HandLandmark


def lm(x: float, y: float, z: float = 0.0) -> HandLandmark:
    return HandLandmark(x=x, y=y, z=z)


def straight_finger(
    base_x: float, base_y: float, dx: float, dy: float, dz: float = 0.0,
) -> List[HandLandmark]:
    """Four colinear landmarks from (base_x, base_y) along (dx, dy, dz).

    Used wherever a test needs an "extended" finger. The PIP angle is
    exactly 180°, so the extension score saturates to 1.0.
    """
    return [
        lm(base_x + dx * t, base_y + dy * t, dz * t)
        for t in (0.0, 0.33, 0.66, 1.0)
    ]


def curled_finger(
    base_x: float, base_y: float, fold_dy: float = 0.05,
) -> List[HandLandmark]:
    """Finger curled so TIP returns to the MCP position.

    MCP at (x, y), PIP offset to (x, y - fold_dy), TIP back at (x, y).
    v1 = (MCP - PIP) and v2 = (TIP - PIP) point in the same direction
    (cos = +1), so the extension score saturates to 0.0.
    """
    return [
        lm(base_x, base_y),                 # MCP
        lm(base_x, base_y - fold_dy),       # PIP
        lm(base_x, base_y - fold_dy / 2),   # DIP (anywhere reasonable)
        lm(base_x, base_y),                 # TIP back at MCP
    ]


def build_hand(
    *,
    wrist: Tuple[float, float, float] = (0.50, 0.90, 0.0),
    thumb: Sequence[HandLandmark],
    index: Sequence[HandLandmark],
    middle: Sequence[HandLandmark],
    ring: Sequence[HandLandmark],
    pinky: Sequence[HandLandmark],
    handedness: str = "Right",
    handedness_score: float = 0.99,
) -> Hand:
    """Assemble a 21-landmark Hand from the wrist + 5 four-point fingers."""
    landmarks = (lm(*wrist), *thumb, *index, *middle, *ring, *pinky)
    if len(landmarks) != 21:
        raise ValueError(
            f"expected 21 landmarks total, got {len(landmarks)} — "
            f"each finger needs exactly 4 landmarks"
        )
    return Hand(
        landmarks=landmarks,
        handedness=handedness,
        handedness_score=handedness_score,
    )


# ----- canonical poses (return new Hand instances; Hand is frozen) -----


def open_palm_hand() -> Hand:
    """All five fingers straight."""
    return build_hand(
        thumb=straight_finger(0.40, 0.85, -0.15, -0.10),
        index=straight_finger(0.45, 0.75, -0.02, -0.40),
        middle=straight_finger(0.50, 0.75,  0.00, -0.45),
        ring=straight_finger(0.55, 0.75,   0.02, -0.40),
        pinky=straight_finger(0.60, 0.78,  0.10, -0.30),
    )


def fist_hand() -> Hand:
    """Four non-thumb fingers curled; thumb left semi-straight to mimic a
    naturally wrapped fist (real-world thumbs don't fully tuck)."""
    return build_hand(
        thumb=straight_finger(0.40, 0.85, -0.05, -0.03),
        index=curled_finger(0.45, 0.75),
        middle=curled_finger(0.50, 0.75),
        ring=curled_finger(0.55, 0.75),
        pinky=curled_finger(0.60, 0.78),
    )


def three_down_hand(thumb_index_gap: float = 0.10) -> Hand:
    """Volume-slider pose: middle/ring/pinky curled, thumb and index
    extended with a configurable gap between their tips.

    `thumb_index_gap` controls the horizontal separation between thumb
    tip and index tip in normalized image space. Larger gap = "louder"
    in VolumeAction's mapping.
    """
    # Index extends straight up from its MCP.
    index_tip_x = 0.45
    # Thumb angles outward; controlling its horizontal offset controls
    # the thumb-to-index gap.
    thumb_tip_x = index_tip_x - thumb_index_gap
    thumb = [
        lm(0.40, 0.85),
        lm(thumb_tip_x + 0.05, 0.75),
        lm(thumb_tip_x + 0.025, 0.65),
        lm(thumb_tip_x, 0.55),
    ]
    index = straight_finger(0.45, 0.75, 0.0, -0.40)
    return build_hand(
        thumb=thumb,
        index=index,
        middle=curled_finger(0.50, 0.75),
        ring=curled_finger(0.55, 0.75),
        pinky=curled_finger(0.60, 0.78),
    )


def pinch_hand() -> Hand:
    """Thumb tip meeting index tip at a single point.

    The thumb extends mostly straight; the index has a slight bend at
    PIP so it can curl to meet the thumb (index extension lands around
    0.7-0.8 — semi-extended, the regime PinchDetector requires).
    Middle/ring/pinky stay extended (OK-sign style).
    """
    touch = (0.50, 0.55, 0.0)
    thumb = [lm(0.40, 0.85), lm(0.43, 0.75), lm(0.47, 0.65), lm(*touch)]
    index = [lm(0.45, 0.75), lm(0.45, 0.60), lm(0.48, 0.55), lm(*touch)]
    return build_hand(
        thumb=thumb,
        index=index,
        middle=straight_finger(0.50, 0.75, 0.00, -0.45),
        ring=straight_finger(0.55, 0.75,  0.02, -0.40),
        pinky=straight_finger(0.60, 0.78,  0.10, -0.30),
    )


def two_fingers_hand(thumb_up: bool = False) -> Hand:
    """Index + middle extended, ring + pinky curled.

    thumb_up toggles whether the thumb is extended out to the side (True)
    or tucked across the palm (False) — both are valid two-finger poses
    and the detector must accept both.
    """
    if thumb_up:
        thumb = straight_finger(0.40, 0.85, -0.20, -0.05)  # sticks out
    else:
        thumb = curled_finger(0.40, 0.85)                  # tucked
    return build_hand(
        thumb=thumb,
        index=straight_finger(0.45, 0.75, -0.02, -0.40),
        middle=straight_finger(0.50, 0.75, 0.00, -0.45),
        ring=curled_finger(0.55, 0.75),
        pinky=curled_finger(0.60, 0.78),
    )


def scaled(hand: Hand, factor: float, center: Tuple[float, float] = (0.5, 0.5)) -> Hand:
    """Return a new Hand with all landmarks scaled by `factor` about `center`.

    Used for scale-invariance tests: if a detector depends on raw distances
    instead of hand_scale-normalized ones, scaling the hand will break it.
    """
    cx, cy = center
    new_landmarks = tuple(
        lm(cx + (p.x - cx) * factor, cy + (p.y - cy) * factor, p.z * factor)
        for p in hand.landmarks
    )
    return Hand(
        landmarks=new_landmarks,
        handedness=hand.handedness,
        handedness_score=hand.handedness_score,
    )
