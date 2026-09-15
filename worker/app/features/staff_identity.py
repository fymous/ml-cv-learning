"""Staff identity — a reusable "is this person a staff member?" capability.

This is deliberately NOT an attendance-only helper. Many features need to know
who is staff (attendance, footfall accuracy, dwell/queue exclusion, staff
coverage analytics, understaffing alerts, staff-area breach). So the learned
*staff profile* lives here as a shared, persisted artifact that any feature can
load.

Layer 1 (this module): a learned **uniform colour** model.

  Onboarding: staff stand in an `enroll` zone; we sample their torso colour
  across many frames and build a robust set of HSV ranges — the "staff
  profile" — saved per store as `staff_profile.json`. At run time
  `StaffFilter` loads these ranges and tags staff by uniform *anywhere in
  frame*, not just when they stand in a fixed zone. That fixes the real
  problem: floor/sales staff who roam to help customers are recognised, not
  only the cashier behind the counter.

  Class-level by design — it models "what the uniform looks like", so new hires
  who wear the uniform are recognised automatically with no re-enrollment.

Layer 2 (future): appearance ReID embeddings captured by the same `enroll`
gesture, for robustness against look-alike customers, lighting and ID loss.

Why HSV and a percentile envelope: hue is kept tight (the actual colour) while
value/saturation are allowed to vary, so the same uniform matches in bright
light or shadow. Ranges are built from the 5th–95th percentile of many samples
(padded), which rejects outliers (a customer who briefly stood in the zone)
while covering real lighting spread. Red uniforms wrap around the hue circle
(near 0 and 180), so those are split into two ranges automatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.inference.backend import Detection

logger = logging.getLogger(__name__)

# Quality gates so we only learn from clean samples.
_MIN_CONF = 0.45
_MIN_BBOX_H = 70  # px — ignore tiny/far people
# Padding added around the measured percentile envelope, per HSV channel.
_PAD_H, _PAD_S, _PAD_V = 8, 50, 60
_HSV_MAX = np.array([179, 255, 255])
_HSV_MIN = np.array([0, 0, 0])


def torso_hsv_median(frame_bgr, det: Detection) -> np.ndarray | None:
    """Median HSV colour of a person's torso patch, or None if unusable.

    Same central-torso ROI used by StaffFilter's uniform match (15%–55% down,
    middle 40% across) so learning and matching agree.
    """
    try:
        import cv2

        h, w = frame_bgr.shape[:2]
        bw, bh = det.x2 - det.x1, det.y2 - det.y1
        tx1 = int(max(0, det.x1 + 0.30 * bw))
        tx2 = int(min(w, det.x2 - 0.30 * bw))
        ty1 = int(max(0, det.y1 + 0.15 * bh))
        ty2 = int(min(h, det.y1 + 0.55 * bh))
        if tx2 <= tx1 or ty2 <= ty1:
            return None
        roi = frame_bgr[ty1:ty2, tx1:tx2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return np.median(hsv.reshape(-1, 3), axis=0)
    except Exception:  # noqa: BLE001 — never break enrollment on a bad crop
        return None


def is_sample_worthy(det: Detection) -> bool:
    return (
        det.class_name == "person"
        and det.confidence >= _MIN_CONF
        and (det.y2 - det.y1) >= _MIN_BBOX_H
    )


def profile_coverage(uniform_hsv: list[list[int]]) -> float:
    """Fraction of HSV colour-space volume the learned ranges cover (0–1).

    A real uniform is one consistent colour → small coverage. If people in the
    enroll zone wore many different colours (i.e. there is no real uniform, or
    customers wandered into the zone), the percentile envelope balloons and
    coverage is large — a signal the profile is unreliable.
    """
    total = 0.0
    for r in uniform_hsv:
        if len(r) != 6:
            continue
        hs = max(0, r[3] - r[0]) / 179.0
        ss = max(0, r[4] - r[1]) / 255.0
        vs = max(0, r[5] - r[2]) / 255.0
        total += hs * ss * vs
    return min(1.0, total)


def uniform_spread_warning(uniform_hsv: list[list[int]], threshold: float = 0.12) -> str | None:
    """Human-readable warning if the learned uniform is too broad to be trusted."""
    cov = profile_coverage(uniform_hsv)
    if cov > threshold:
        return (
            f"The learned uniform covers {cov * 100:.0f}% of the colour space — too "
            "broad to be a single uniform. The people in the enroll zone were not "
            "wearing one consistent colour (likely no real uniform in this clip, or "
            "mixed customers). Staff-by-uniform will over-tag. Re-run onboarding with "
            "uniformed staff only in the zone, or rely on staff zones / ReID."
        )
    return None


def range_mid_bgr(hsv_range: list[int]) -> tuple[int, int, int]:
    """Representative BGR colour for a learned HSV range (for UI swatches)."""
    try:
        import cv2

        h = (hsv_range[0] + hsv_range[3]) / 2
        s = (hsv_range[1] + hsv_range[4]) / 2
        v = max(120, (hsv_range[2] + hsv_range[5]) / 2)  # brighten so it reads
        px = np.uint8([[[h, s, v]]])
        b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0].tolist()
        return int(b), int(g), int(r)
    except Exception:  # noqa: BLE001
        return 128, 128, 128


@dataclass
class StaffProfile:
    """The learned, persisted answer to 'what does staff look like here'."""

    uniform_hsv: list[list[int]] = field(default_factory=list)
    sample_count: int = 0
    tracks_sampled: int = 0
    created_at: str = ""
    source: str = ""
    version: int = 1

    @property
    def enabled(self) -> bool:
        return bool(self.uniform_hsv)

    def to_dict(self) -> dict:
        return {
            "uniform_hsv": self.uniform_hsv,
            "sample_count": self.sample_count,
            "tracks_sampled": self.tracks_sampled,
            "created_at": self.created_at,
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StaffProfile":
        return cls(
            uniform_hsv=[list(map(int, r)) for r in d.get("uniform_hsv", []) if len(r) == 6],
            sample_count=int(d.get("sample_count", 0)),
            tracks_sampled=int(d.get("tracks_sampled", 0)),
            created_at=str(d.get("created_at", "")),
            source=str(d.get("source", "")),
            version=int(d.get("version", 1)),
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "StaffProfile | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            logger.exception("failed to load staff profile %s", path)
            return None


class UniformLearner:
    """Accumulates torso-colour samples during onboarding and builds a
    StaffProfile (one or two robust HSV ranges)."""

    def __init__(self, source: str = "") -> None:
        self._samples: list[np.ndarray] = []
        self._track_ids: set[int] = set()
        self._source = source

    @property
    def n(self) -> int:
        return len(self._samples)

    def add_sample(self, frame_bgr, det: Detection) -> bool:
        if not is_sample_worthy(det):
            return False
        med = torso_hsv_median(frame_bgr, det)
        if med is None:
            return False
        self._samples.append(med)
        if det.track_id is not None:
            self._track_ids.add(det.track_id)
        return True

    def _envelope(self, sub: np.ndarray) -> list[int]:
        lo = np.percentile(sub, 5, axis=0) - np.array([_PAD_H, _PAD_S, _PAD_V])
        hi = np.percentile(sub, 95, axis=0) + np.array([_PAD_H, _PAD_S, _PAD_V])
        lo = np.clip(lo, _HSV_MIN, _HSV_MAX).astype(int)
        hi = np.clip(hi, _HSV_MIN, _HSV_MAX).astype(int)
        return [int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2])]

    def build(self) -> StaffProfile:
        ranges: list[list[int]] = []
        if self._samples:
            hsv = np.array(self._samples)
            hue = hsv[:, 0]
            # Red wraps the hue circle: samples cluster near 0 AND near 180.
            wrap = bool((hue < 15).any() and (hue > 165).any())
            if wrap:
                low = hsv[hue < 90]
                high = hsv[hue >= 90]
                if len(low):
                    ranges.append(self._envelope(low))
                if len(high):
                    ranges.append(self._envelope(high))
            else:
                ranges.append(self._envelope(hsv))
        return StaffProfile(
            uniform_hsv=ranges,
            sample_count=len(self._samples),
            tracks_sampled=len(self._track_ids),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=self._source,
        )
