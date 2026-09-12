"""Entrance counting — directional, per-track crossings (not alerts).

Two modes per `entrance` zone:

1. **Line crossing (preferred)** — the zone carries a directed tripwire `line`
   ([[ax, ay], [bx, by]]). A track is counted when its feet-point *trajectory*
   crosses that line. Direction (enter vs exit) comes from the movement vector
   vs the line's inside normal (or the zone's `in_direction`). This fixes the
   classic polygon bug where a person who steps in and straight back out is
   counted twice — a bounce never completes an inward crossing.

2. **Polygon presence (legacy fallback)** — if the zone has no `line`, we keep
   the old outside->inside (and first-sight-inside) behaviour.

Both modes:
- de-duplicate same-frame boxes with NMS,
- guard against ByteTrack/BoT-SORT ID splits,
- debounce so one track cannot re-count within a short cooldown,
- require a minimum track age before the first count,
- emit an approximate DwellEvent once a counted track disappears.

`CrossingEvent` keeps the exact fields demographics/group already consume;
`direction` is now "enter" or "exit".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.inference.backend import Detection
from app.models import ZoneConfig
from app.zones import (
    crossing_is_inward,
    line_to_frame_space,
    point_in_polygon,
    polygon_to_frame_space,
    segments_intersect,
)

logger = logging.getLogger(__name__)

_MISS_GRACE_FRAMES = 12
_NMS_IOU = 0.40
_SPLIT_IOU = 0.50
# A track must be seen at least this many frames before it can count. Kills the
# "brand new ID appears already across the line" spurious count.
_MIN_TRACK_FRAMES = 2
# Same track cannot register another crossing within this many seconds.
_COUNT_COOLDOWN_SEC = 1.0


@dataclass
class CrossingEvent:
    direction: str  # "enter" | "exit"
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
    frames: int = 0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    fx: float = 0.0  # last feet-point x
    fy: float = 0.0  # last feet-point y
    has_feet: bool = False
    entered_seq: int | None = None
    entered_at: float = 0.0
    last_seen_at: float = 0.0
    last_count_at: float = -1e9


@dataclass
class ZoneFootfall:
    zone_id: str
    zone_name: str
    entered: int = 0
    exited: int = 0
    _tracks: dict[int, _TrackState] = field(default_factory=dict, repr=False)
    _dwell_events: list[DwellEvent] = field(default_factory=list, repr=False)

    @property
    def net(self) -> int:
        return self.entered - self.exited

    def as_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "entered": self.entered,
            "exited": self.exited,
            "net": self.net,
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
        exclude_ids: set[int] | None = None,
    ) -> list[CrossingEvent]:
        """Count crossings. `exclude_ids` (e.g. staff/guards) are tracked for
        state but never counted or emitted as events."""
        events: list[CrossingEvent] = []
        if not self._zones:
            return events
        exclude = exclude_ids or set()
        persons = _nms_persons(
            [d for d in detections if d.class_name == "person" and d.track_id is not None],
            _NMS_IOU,
        )
        for zf in self._zones:
            cfg = self._zone_cfg[zf.zone_id]
            if cfg.line:
                self._update_line_zone(zf, cfg, persons, frame_w, frame_h, media_ts, events, exclude)
            else:
                self._update_polygon_zone(zf, cfg, persons, frame_w, frame_h, media_ts, events, exclude)
        return events

    # ------------------------------------------------------------------ line
    def _update_line_zone(
        self,
        zf: ZoneFootfall,
        cfg: ZoneConfig,
        persons: list[Detection],
        frame_w: int,
        frame_h: int,
        media_ts: float,
        events: list[CrossingEvent],
        exclude: set[int],
    ) -> None:
        (ax, ay), (bx, by) = line_to_frame_space(cfg.line, frame_w, frame_h)
        a, b = (ax, ay), (bx, by)
        in_dir = tuple(cfg.in_direction) if cfg.in_direction else None
        seen: set[int] = set()
        for d in persons:
            tid = d.track_id
            assert tid is not None
            seen.add(tid)
            fx, fy = _feet_point(d)
            st = zf._tracks.get(tid)
            if st is None:
                zf._tracks[tid] = _TrackState(
                    inside=False, frames=1, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                    fx=fx, fy=fy, has_feet=True, last_seen_at=media_ts,
                )
                continue
            st.frames += 1
            st.missed = 0
            st.x1, st.y1, st.x2, st.y2 = d.x1, d.y1, d.x2, d.y2
            st.last_seen_at = media_ts

            if (
                tid not in exclude
                and st.has_feet
                and st.frames >= _MIN_TRACK_FRAMES
                and (media_ts - st.last_count_at) >= _COUNT_COOLDOWN_SEC
                and segments_intersect((st.fx, st.fy), (fx, fy), a, b)
            ):
                inward = crossing_is_inward((st.fx, st.fy), (fx, fy), a, b, in_dir)
                if inward:
                    zf.entered += 1
                    st.entered_seq = zf.entered
                    st.entered_at = media_ts
                    st.inside = True
                    events.append(self._event("enter", zf, tid, zf.entered, d))
                    logger.info(
                        "footfall ENTER (line) zone=%s track=%s entered=%d net=%d t=%.1f",
                        zf.zone_id, tid, zf.entered, zf.net, media_ts,
                    )
                else:
                    zf.exited += 1
                    st.inside = False
                    events.append(self._event("exit", zf, tid, zf.exited, d))
                    logger.info(
                        "footfall EXIT (line) zone=%s track=%s exited=%d net=%d t=%.1f",
                        zf.zone_id, tid, zf.exited, zf.net, media_ts,
                    )
                st.last_count_at = media_ts

            st.fx, st.fy = fx, fy
            st.has_feet = True

        self._age_out(zf, seen)

    # --------------------------------------------------------------- polygon
    def _update_polygon_zone(
        self,
        zf: ZoneFootfall,
        cfg: ZoneConfig,
        persons: list[Detection],
        frame_w: int,
        frame_h: int,
        media_ts: float,
        events: list[CrossingEvent],
        exclude: set[int],
    ) -> None:
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
                if (
                    inside
                    and tid not in exclude
                    and not _is_id_split(zf._tracks, tid, d.x1, d.y1, d.x2, d.y2)
                ):
                    zf.entered += 1
                    zf._tracks[tid] = _TrackState(
                        inside=True, frames=1, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                        fx=fx, fy=fy, has_feet=True, entered_seq=zf.entered,
                        entered_at=media_ts, last_seen_at=media_ts, last_count_at=media_ts,
                    )
                    events.append(self._event("enter", zf, tid, zf.entered, d))
                    logger.info(
                        "footfall ENTER (first-sight) zone=%s track=%s entered=%d t=%.1f",
                        zf.zone_id, tid, zf.entered, media_ts,
                    )
                else:
                    zf._tracks[tid] = _TrackState(
                        inside=inside, frames=1, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                        fx=fx, fy=fy, has_feet=True, last_seen_at=media_ts,
                    )
                continue
            prev = state.inside
            state.frames += 1
            state.missed = 0
            state.x1, state.y1, state.x2, state.y2 = d.x1, d.y1, d.x2, d.y2
            state.last_seen_at = media_ts
            if (
                not prev
                and inside
                and tid not in exclude
                and (media_ts - state.last_count_at) >= _COUNT_COOLDOWN_SEC
                and not _is_id_split(zf._tracks, tid, d.x1, d.y1, d.x2, d.y2)
            ):
                zf.entered += 1
                state.entered_seq = zf.entered
                state.entered_at = media_ts
                state.last_count_at = media_ts
                events.append(self._event("enter", zf, tid, zf.entered, d))
                logger.info(
                    "footfall ENTER zone=%s track=%s entered=%d t=%.1f",
                    zf.zone_id, tid, zf.entered, media_ts,
                )
            state.inside = inside
            state.fx, state.fy = fx, fy
            state.has_feet = True

        self._age_out(zf, seen)

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _event(
        direction: str, zf: ZoneFootfall, tid: int, seq_n: int, d: Detection
    ) -> CrossingEvent:
        return CrossingEvent(
            direction=direction,
            zone_id=zf.zone_id,
            zone_name=zf.zone_name,
            track_id=tid,
            seq_n=seq_n,
            x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
        )

    @staticmethod
    def _age_out(zf: ZoneFootfall, seen: set[int]) -> None:
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
