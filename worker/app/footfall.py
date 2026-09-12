"""Entrance counting — unique track crossings (not alerts).

Counts *entered* when a track:
  - moves outside → inside an `entrance` zone (feet point), OR
  - is first seen already inside the zone (handles low-fps / camera-cut cases).

Still tracks inside/outside so a person can leave and re-enter.

Also emits an approximate DwellEvent per counted entry once the track is no
longer seen anywhere in frame. Additive — never affects the entered count.

Duplicate-box suppression:
- NMS (same frame): drops a weaker overlapping detection of the same body.
- ID-split guard (cross-frame): when a new enter fires, check if any
  currently-inside track has a heavily overlapping box (IoU >= _SPLIT_IOU).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.inference.backend import Detection
from app.models import ZoneConfig
from app.zones import point_in_polygon, polygon_to_frame_space

logger = logging.getLogger(__name__)

_MISS_GRACE_FRAMES = 12
_NMS_IOU = 0.40
_SPLIT_IOU = 0.50


@dataclass
class CrossingEvent:
    direction: str  # "enter"
    zone_id: str
    zone_name: str
    track_id: int
    seq_n: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class DwellEvent:
    zone_id: str
    track_id: int
    entered_seq: int
    dwell_sec: float


@dataclass
class _TrackState:
    inside: bool
    missed: int = 0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    entered_seq: int | None = None
    entered_at: float = 0.0
    last_seen_at: float = 0.0


@dataclass
class ZoneFootfall:
    zone_id: str
    zone_name: str
    entered: int = 0
    _tracks: dict[int, _TrackState] = field(default_factory=dict, repr=False)
    _dwell_events: list[DwellEvent] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "entered": self.entered,
            "exited": 0,
            "net": self.entered,
        }


def _feet_point(d: Detection) -> tuple[float, float]:
    cx = (d.x1 + d.x2) / 2
    h = max(1.0, d.y2 - d.y1)
    cy = d.y2 - 0.08 * h
    return cx, cy


def _iou(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _is_id_split(
    tracks: dict[int, _TrackState],
    new_tid: int,
    x1: float, y1: float, x2: float, y2: float,
) -> bool:
    for tid, st in tracks.items():
        if tid == new_tid or not st.inside:
            continue
        if _iou(x1, y1, x2, y2, st.x1, st.y1, st.x2, st.y2) >= _SPLIT_IOU:
            return True
    return False


def _nms_persons(persons: list[Detection], iou_thresh: float) -> list[Detection]:
    ordered = sorted(persons, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for d in ordered:
        if any(d.iou(k) >= iou_thresh for k in kept):
            continue
        kept.append(d)
    return kept


class EntranceCounter:
    def __init__(self, zones: list[ZoneConfig], zone_names: dict[str, str] | None = None) -> None:
        names = zone_names or {}
        self._zones = [
            ZoneFootfall(
                zone_id=z.zone_id,
                zone_name=names.get(z.zone_id, z.name or z.zone_type),
            )
            for z in zones
            if z.zone_type == "entrance"
        ]
        self._zone_cfg = {z.zone_id: z for z in zones if z.zone_type == "entrance"}

    @property
    def enabled(self) -> bool:
        return bool(self._zones)

    def update(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        media_ts: float = 0.0,
    ) -> list[CrossingEvent]:
        events: list[CrossingEvent] = []
        if not self._zones:
            return events
        persons = _nms_persons(
            [d for d in detections if d.class_name == "person" and d.track_id is not None],
            _NMS_IOU,
        )
        for zf in self._zones:
            cfg = self._zone_cfg[zf.zone_id]
            poly = polygon_to_frame_space(cfg.polygon, frame_w, frame_h)
            seen: set[int] = set()
            for d in persons:
                tid = d.track_id
                assert tid is not None
                seen.add(tid)
                fx, fy = _feet_point(d)
                inside = point_in_polygon(fx, fy, poly)
                state = zf._tracks.get(tid)
                if state is None:
                    if inside:
                        if _is_id_split(zf._tracks, tid, d.x1, d.y1, d.x2, d.y2):
                            logger.debug(
                                "footfall ID-split suppressed (first-sight) zone=%s track=%s t=%.1f",
                                zf.zone_id, tid, media_ts,
                            )
                            zf._tracks[tid] = _TrackState(
                                inside=True, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2
                            )
                        else:
                            zf.entered += 1
                            zf._tracks[tid] = _TrackState(
                                inside=True, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                                entered_seq=zf.entered, entered_at=media_ts, last_seen_at=media_ts,
                            )
                            events.append(
                                CrossingEvent(
                                    direction="enter",
                                    zone_id=zf.zone_id,
                                    zone_name=zf.zone_name,
                                    track_id=tid,
                                    seq_n=zf.entered,
                                    x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                                )
                            )
                            logger.info(
                                "footfall ENTER (first-sight) zone=%s track=%s total_entered=%d t=%.1f",
                                zf.zone_id, tid, zf.entered, media_ts,
                            )
                    else:
                        zf._tracks[tid] = _TrackState(
                            inside=False, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2
                        )
                    continue
                prev = state.inside
                state.missed = 0
                state.x1, state.y1, state.x2, state.y2 = d.x1, d.y1, d.x2, d.y2
                state.last_seen_at = media_ts
                if not prev and inside:
                    if _is_id_split(zf._tracks, tid, d.x1, d.y1, d.x2, d.y2):
                        logger.debug(
                            "footfall ID-split suppressed zone=%s track=%s t=%.1f",
                            zf.zone_id, tid, media_ts,
                        )
                    else:
                        zf.entered += 1
                        state.entered_seq = zf.entered
                        state.entered_at = media_ts
                        events.append(
                            CrossingEvent(
                                direction="enter",
                                zone_id=zf.zone_id,
                                zone_name=zf.zone_name,
                                track_id=tid,
                                seq_n=zf.entered,
                                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                            )
                        )
                        logger.info(
                            "footfall ENTER zone=%s track=%s total_entered=%d t=%.1f",
                            zf.zone_id, tid, zf.entered, media_ts,
                        )
                state.inside = inside
            for tid in list(zf._tracks.keys()):
                if tid in seen:
                    continue
                st = zf._tracks[tid]
                st.missed += 1
                if st.missed > _MISS_GRACE_FRAMES:
                    if st.entered_seq is not None:
                        zf._dwell_events.append(
                            DwellEvent(
                                zone_id=zf.zone_id,
                                track_id=tid,
                                entered_seq=st.entered_seq,
                                dwell_sec=max(0.0, st.last_seen_at - st.entered_at),
                            )
                        )
                    del zf._tracks[tid]
        return events

    def summary(self) -> list[dict]:
        return [z.as_dict() for z in self._zones]

    def pop_dwell_events(self) -> list[DwellEvent]:
        events: list[DwellEvent] = []
        for zf in self._zones:
            if zf._dwell_events:
                events.extend(zf._dwell_events)
                zf._dwell_events = []
        return events

    def finalize(self) -> list[DwellEvent]:
        events: list[DwellEvent] = []
        for zf in self._zones:
            for tid, st in list(zf._tracks.items()):
                if st.entered_seq is not None:
                    events.append(
                        DwellEvent(
                            zone_id=zf.zone_id,
                            track_id=tid,
                            entered_seq=st.entered_seq,
                            dwell_sec=max(0.0, st.last_seen_at - st.entered_at),
                        )
                    )
            zf._tracks.clear()
        events.extend(self.pop_dwell_events())
        return events
