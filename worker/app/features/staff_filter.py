"""Staff / passer-by exclusion — decide which tracks must NOT be counted.

People count should reflect genuine visitors, not the security guard at the
door or staff milling around the entrance. This module tags `track_id`s to
exclude, and `EntranceCounter` skips counting them.

Two cheap, local mechanisms (no extra heavy model):

1. **Staff / exclusion zone** (`zone_type: "staff"` or `"exclude"`). Any track
   whose feet enter that polygon is tagged staff *from then on*. Draw it over
   the guard's post / patrol spot or a staff-only area. Deterministic and the
   most reliable option here.

2. **Uniform colour** (optional, off unless `STAFF_UNIFORM_HSV` is set). On a
   fresh track, sample the torso colour; if it matches a configured uniform
   HSV range, tag staff. Helps for guards who move around, but is a heuristic
   (a visitor in a similar colour can false-match) — use as a supplement.

Note: robustly excluding a specific person regardless of position/clothing
needs appearance ReID enrollment (enroll each guard once). That is a future
`features/reid.py`, not implemented here.

Tags are sticky per track: once staff, always staff for that `track_id`.
"""

from __future__ import annotations

import logging

from app.inference.backend import Detection
from app.models import ZoneConfig
from app.zones import point_in_polygon, polygon_to_frame_space

logger = logging.getLogger(__name__)

_STAFF_ZONE_TYPES = ("staff", "exclude")


def _feet_point(d: Detection) -> tuple[float, float]:
    cx = (d.x1 + d.x2) / 2
    h = max(1.0, d.y2 - d.y1)
    cy = d.y2 - 0.08 * h
    return cx, cy


class StaffFilter:
    def __init__(
        self,
        zones: list[ZoneConfig],
        uniform_hsv: list[list[int]] | None = None,
    ) -> None:
        self._staff_zones = [z for z in zones if z.zone_type in _STAFF_ZONE_TYPES]
        # each range: [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi] in OpenCV HSV
        self._uniform = [list(r) for r in (uniform_hsv or []) if len(r) == 6]
        self._staff_ids: set[int] = set()

    @property
    def enabled(self) -> bool:
        return bool(self._staff_zones) or bool(self._uniform)

    def mark(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        frame=None,
    ) -> set[int]:
        """Update and return the cumulative set of staff/excluded track ids."""
        if not self.enabled:
            return self._staff_ids
        polys = [
            polygon_to_frame_space(z.polygon, frame_w, frame_h)
            for z in self._staff_zones
            if z.polygon
        ]
        for d in detections:
            tid = d.track_id
            if tid is None or d.class_name != "person" or tid in self._staff_ids:
                continue
            fx, fy = _feet_point(d)
            if any(point_in_polygon(fx, fy, poly) for poly in polys):
                self._staff_ids.add(tid)
                logger.info("staff TAG (zone) track=%s", tid)
                continue
            if self._uniform and frame is not None and self._uniform_match(frame, d):
                self._staff_ids.add(tid)
                logger.info("staff TAG (uniform) track=%s", tid)
        return self._staff_ids

    def _uniform_match(self, frame, d: Detection) -> bool:
        """True if the person's torso colour falls in a configured uniform range.

        Fail-soft: any error (no cv2, bad crop) returns False.
        """
        try:
            import cv2
            import numpy as np

            h, w = frame.shape[:2]
            bw, bh = d.x2 - d.x1, d.y2 - d.y1
            # central torso ROI: 15%–55% down, middle 40% across
            tx1 = int(max(0, d.x1 + 0.30 * bw))
            tx2 = int(min(w, d.x2 - 0.30 * bw))
            ty1 = int(max(0, d.y1 + 0.15 * bh))
            ty2 = int(min(h, d.y1 + 0.55 * bh))
            if tx2 <= tx1 or ty2 <= ty1:
                return False
            roi = frame[ty1:ty2, tx1:tx2]
            if roi.size == 0:
                return False
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            med = np.median(hsv.reshape(-1, 3), axis=0)
            for h_lo, s_lo, v_lo, h_hi, s_hi, v_hi in self._uniform:
                if (
                    h_lo <= med[0] <= h_hi
                    and s_lo <= med[1] <= s_hi
                    and v_lo <= med[2] <= v_hi
                ):
                    return True
            return False
        except Exception:  # noqa: BLE001 — never break the loop on colour check
            return False

    def summary(self) -> dict:
        return {"staff_excluded": len(self._staff_ids)}
