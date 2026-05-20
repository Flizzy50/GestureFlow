"""Pure-function feature extraction from a Hand.

Design contract for this module:
  - Every function takes a Hand (and only a Hand) and returns a value.
  - No I/O, no state, no logging, no mutation.
  - That makes everything here trivially unit-testable with fabricated
    Hand objects, and lets the recognizer compute features once per frame
    and share them across every detector.

The math is in MediaPipe's normalized image space: x and y are in [0, 1]
relative to image width/height, z is depth relative to the wrist. We treat
all three as a 3D vector for geometry — using z makes finger-extension
detection robust to in-plane hand rotation that 2D would confuse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from vision.hand_tracker import Hand, HandLandmark


class LM:
    """Named landmark indices for MediaPipe's 21-point hand topology.

    Readable names beat magic numbers everywhere in this codebase.
    Anatomy reference:
      MCP = metacarpophalangeal (knuckle)
      PIP = proximal interphalangeal (middle joint of a finger)
      DIP = distal interphalangeal (joint nearest the tip)
      IP  = interphalangeal (the single joint in the thumb's middle)
    """
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


# (base, middle-joint, tip) triples used for the angle-based extension
# test. Index/middle/ring/pinky bend at the PIP joint. The thumb bends
# at the IP joint, so it gets its own triple.
_FINGER_JOINTS: Tuple[Tuple[int, int, int], ...] = (
    (LM.INDEX_MCP,  LM.INDEX_PIP,  LM.INDEX_TIP),
    (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP),
    (LM.RING_MCP,   LM.RING_PIP,   LM.RING_TIP),
    (LM.PINKY_MCP,  LM.PINKY_PIP,  LM.PINKY_TIP),
)
_THUMB_JOINT: Tuple[int, int, int] = (LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP)


Vec3 = Tuple[float, float, float]


def _vec(a: HandLandmark, b: HandLandmark) -> Vec3:
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def _dot(u: Vec3, v: Vec3) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def distance(a: HandLandmark, b: HandLandmark) -> float:
    """Euclidean distance between two landmarks in normalized image space."""
    return _norm(_vec(a, b))


def hand_scale(hand: Hand) -> float:
    """Wrist -> middle-finger MCP distance. A flex-invariant ruler.

    Why this measurement: it's the longest internal hand segment that does
    NOT change as fingers curl. Using it to normalize other distances (like
    pinch's thumb-to-index gap) makes detection work at any distance from
    the camera — without it, pinch would only fire at one specific zoom
    level. This is the kind of detail that separates a real CV system from
    a hardcoded prototype.
    """
    return distance(hand.landmark(LM.WRIST), hand.landmark(LM.MIDDLE_MCP))


def pinch_distance(hand: Hand) -> float:
    """Thumb-tip to index-tip distance normalized by hand scale.

    Returns a unitless ratio:
      ~0.15 = fingers touching (pinched)
      ~0.35 = roughly "ok sign" gap
      ~0.70+ = open hand spread

    Returns +inf for degenerate hands with zero scale (shouldn't happen
    from real MediaPipe output, but guard anyway).
    """
    raw = distance(hand.landmark(LM.THUMB_TIP), hand.landmark(LM.INDEX_TIP))
    scale = hand_scale(hand)
    if scale < 1e-6:
        return float("inf")
    return raw / scale


@dataclass(frozen=True)
class FingerExtension:
    """Continuous [0, 1] extension score for each finger.

    1.0 = fully straight, 0.0 = fully curled toward the palm.
    Smooth scores (not booleans) so confidences compose cleanly downstream
    and Phase 5's diagnostic overlay has something interesting to draw.
    """
    thumb: float
    index: float
    middle: float
    ring: float
    pinky: float

    @property
    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def count_extended(self, threshold: float = 0.7) -> int:
        """How many fingers cross the 'extended' threshold (default 0.7)."""
        return sum(1 for v in self.as_tuple if v >= threshold)


def _joint_extension(hand: Hand, base: int, joint: int, tip: int) -> float:
    """Continuous extension score from the angle at the finger's joint.

    Geometry:
      Build vectors v1 = (joint -> base) and v2 = (joint -> tip).
      - Straight finger: v1 and v2 point in opposite directions, cos ≈ -1.
      - Bent finger:     v1 and v2 point similar directions, cos ≈ +1.
    We map cos linearly from [-1, +0.3] onto extension [1, 0] and clamp:
      cos = -1.0 (180°)  -> extension 1.00
      cos =  0.0 ( 90°)  -> extension 0.23
      cos = +0.3         -> extension 0.00
    The +0.3 upper bound (rather than +1.0) makes the score saturate to 0
    well before the finger is fully closed — useful so a half-bent finger
    reads as "not extended" rather than "0.5 extended", which matches how
    a human would describe it and gives sharper gesture boundaries.

    Working in 3D (x, y, z) instead of just (x, y) makes this robust to
    hand rotation toward/away from the camera.
    """
    b = hand.landmark(base)
    j = hand.landmark(joint)
    t = hand.landmark(tip)
    v1 = _vec(j, b)
    v2 = _vec(j, t)
    n1, n2 = _norm(v1), _norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = _dot(v1, v2) / (n1 * n2)
    score = (0.3 - cos_angle) / 1.3
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def finger_extensions(hand: Hand) -> FingerExtension:
    """Compute the 5-finger extension scores in one pass."""
    idx, mid, rng, pky = (
        _joint_extension(hand, b, j, t) for b, j, t in _FINGER_JOINTS
    )
    thumb = _joint_extension(hand, *_THUMB_JOINT)
    return FingerExtension(
        thumb=thumb, index=idx, middle=mid, ring=rng, pinky=pky,
    )
