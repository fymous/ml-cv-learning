"""Queue occupancy — how many people are inside each `queue` zone this frame.

Does not fire alerts. That is a separate threshold on top of this gauge.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.inference.backend import Detection
from app.models import ZoneConfig
from app.zones import bbox_overlaps_zone, polygon_to_frame_space

_NMS_IOU = 0.40


def _nms_persons(persons: list[Detection], iou_thresh: float) -> list[Detection]:
    ordered = sorted(persons, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for d in ordered:
        if any(d.iou(k) >= iou_thresh for k in kept):
            continue
        kept.append(d)
    return kept


@dataclass
class QueueResult:
    zone_id: str
    zone_name: str
    people_count: int
    peak: int


class QueueMonitor:
    def __init__(self, zones: list[ZoneConfig]) -> None:
        self._zone_cfg = {z.zone_id: z for z in zones if z.zone_type == "queue"}
        self._states: dict[str, QueueResult] = {
            z.zone_id: QueueResult(
                zone_id=z.zone_id,
                zone_name=z.name or "Queue",
                people_count=0,
                peak=0,
            )
            for z in zones
            if z.zone_type == "queue"
        }

    @property
    def enabled(self) -> bool:
        return bool(self._zone_cfg)

    def estimate(
        self, detections: list[Detection], frame_w: int, frame_h: int
    ) -> list[QueueResult]:
        if not self._zone_cfg:
            return []
        persons = _nms_persons(
            [d for d in detections if d.class_name == "person"], _NMS_IOU
        )
        for zone_id, cfg in self._zone_cfg.items():
            poly = polygon_to_frame_space(cfg.polygon, frame_w, frame_h)
            count = sum(
                1 for d in persons if bbox_overlaps_zone(d.x1, d.y1, d.x2, d.y2, poly)
            )
            state = self._states[zone_id]
            state.people_count = count
            state.peak = max(state.peak, count)
        return list(self._states.values())
