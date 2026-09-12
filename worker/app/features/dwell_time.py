"""Dwell time — how long each visitor's track lingers inside a browsing zone.

Deliberately separate from `footfall.py`. Entrance counting answers "did this
person arrive/leave" at a doorway tripwire; dwell time answers a completely
different retail question — "how long did this shopper spend checking
products in the produce aisle / near the counter / in the electronics
section" — measured over its own zone(s), independent of whether footfall is
even enabled.

Zone type: `dwell` (a normal polygon, like `staff`/`queue`/`kit`). Draw as
many as you like — one per aisle/section you care about; each is tracked and
reported separately.

Mechanics, per zone, per frame:
  - a person's feet-point inside the polygon starts (or continues) a timer
    for that `track_id`,
  - brief tracker misses (occlusion, a missed detection) don't reset the
    timer — `_MISS_GRACE_FRAMES` of grace, same idea as footfall's,
  - once a track has been gone (or has stepped out of the zone) longer than
    the grace period, the run is finalized into a `DwellEvent` with the total
    seconds spent, and the state is dropped — but ONLY if the total is at
    least `min_dwell_sec` (default 4.0s, `DWELL_MIN_SEC` env var). Shorter
    stays are silently ignored: no event, no screenshot, not counted at all.
    A brief walk-through of the zone should never be flagged as "dwelling".
  - `finalize(media_ts)` flushes any tracks still inside when the video/run
    ends, so a visitor still browsing at last-frame isn't silently lost.

No extra model, no per-frame inference cost — pure geometry + bookkeeping on
top of detections the pipeline already has. Optionally accepts `exclude_ids`
(e.g. staff tagged by `features/staff_filter.py`) so a staff member restocking
a shelf doesn't get counted as a browsing customer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median

from app.inference.backend import Detection
from app.models import ZoneConfig
from app.zones import point_in_polygon, polygon_to_frame_space

logger = logging.getLogger(__name__)

_ZONE_TYPE = "dwell"

# Grace period (frames of absence/outside-zone) before we finalize a dwell —
# mirrors footfall's occlusion tolerance so a brief tracker blip mid-browse
# doesn't chop one visit into several short ones.
_MISS_GRACE_FRAMES = 15

# Ignore dwells shorter than this — a person's feet clipping the zone edge for
# a frame or two, or just walking through, is noise, not a "visit". Only a
# stay of at least this long is highlighted as a dwell event; shorter ones are
# dropped entirely (no event, no screenshot). Default matches DWELL_MIN_SEC in
# config.py; overridable per-instance via `DwellTracker(min_dwell_sec=...)`.
_MIN_DWELL_SEC = 4.0

# Buckets meaningful for "did they actually engage with this section".
DWELL_BUCKETS: list[tuple[float, float, str]] = [
    (0, 5, "<5s (passed by)"),
    (5, 20, "5-20s (glanced)"),
    (20, 60, "20-60s (looked)"),
    (60, 180, "1-3min (browsed)"),
    (180, 600, "3-10min (engaged)"),
    (600, float("inf"), "10min+ (extended)"),
]


def dwell_bucket(sec: float) -> str:
    for lo, hi, label in DWELL_BUCKETS:
        if lo <= sec < hi:
            return label
    return DWELL_BUCKETS[-1][2]


def _feet_point(d: Detection) -> tuple[float, float]:
    cx = (d.x1 + d.x2) / 2
    h = max(1.0, d.y2 - d.y1)
    cy = d.y2 - 0.08 * h
    return cx, cy


@dataclass
class DwellEvent:
    zone_id: str
    zone_name: str
    track_id: int
    entered_at: float
    left_at: float

    @property
    def dwell_sec(self) -> float:
        return max(0.0, self.left_at - self.entered_at)


@dataclass
class _TrackDwell:
    entered_at: float
    last_seen_at: float
    missed: int = 0


@dataclass
class _ZoneDwell:
    zone_id: str
    zone_name: str
    polygon: list[list[float]]
    _tracks: dict[int, _TrackDwell] = field(default_factory=dict, repr=False)
    _events: list[DwellEvent] = field(default_factory=list, repr=False)


class DwellTracker:
    def __init__(
        self, zones: list[ZoneConfig], min_dwell_sec: float = _MIN_DWELL_SEC
    ) -> None:
        self._zones = [
            _ZoneDwell(zone_id=z.zone_id, zone_name=z.name or z.zone_type, polygon=z.polygon)
            for z in zones
            if z.zone_type == _ZONE_TYPE and z.polygon
        ]
        # Minimum seconds someone must stay before it counts as a dwell at
        # all — a brief pass-through/glance is ignored, not just bucketed low.
        self._min_dwell_sec = min_dwell_sec

    @property
    def enabled(self) -> bool:
        return bool(self._zones)

    def update(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        media_ts: float,
        exclude_ids: set[int] | None = None,
    ) -> list[DwellEvent]:
        """Advance all zone timers by one frame. Returns dwell events that
        just finalized (visitor left / grace expired) this frame."""
        if not self._zones:
            return []
        exclude = exclude_ids or set()
        persons = [
            d for d in detections
            if d.class_name == "person" and d.track_id is not None and d.track_id not in exclude
        ]
        finalized: list[DwellEvent] = []
        for zd in self._zones:
            poly = polygon_to_frame_space(zd.polygon, frame_w, frame_h)
            inside_now: set[int] = set()
            for d in persons:
                tid = d.track_id
                fx, fy = _feet_point(d)
                if not point_in_polygon(fx, fy, poly):
                    continue
                inside_now.add(tid)
                st = zd._tracks.get(tid)
                if st is None:
                    zd._tracks[tid] = _TrackDwell(entered_at=media_ts, last_seen_at=media_ts)
                    logger.info("dwell START zone=%s track=%s t=%.1f", zd.zone_id, tid, media_ts)
                else:
                    st.last_seen_at = media_ts
                    st.missed = 0
            # anyone tracked but not seen inside this frame: age the grace period
            for tid in list(zd._tracks.keys()):
                if tid in inside_now:
                    continue
                st = zd._tracks[tid]
                st.missed += 1
                if st.missed > _MISS_GRACE_FRAMES:
                    ev = self._finalize_one(zd, tid, st)
                    if ev is not None:
                        finalized.append(ev)
        return finalized

    def _finalize_one(self, zd: _ZoneDwell, tid: int, st: _TrackDwell) -> DwellEvent | None:
        del zd._tracks[tid]
        dwell_sec = max(0.0, st.last_seen_at - st.entered_at)
        if dwell_sec < self._min_dwell_sec:
            logger.debug(
                "dwell IGNORED (too short) zone=%s track=%s dwell=%.1fs < min=%.1fs",
                zd.zone_id, tid, dwell_sec, self._min_dwell_sec,
            )
            return None
        ev = DwellEvent(
            zone_id=zd.zone_id, zone_name=zd.zone_name, track_id=tid,
            entered_at=st.entered_at, left_at=st.last_seen_at,
        )
        zd._events.append(ev)
        logger.info(
            "dwell END zone=%s track=%s dwell=%.1fs bucket=%s",
            zd.zone_id, tid, ev.dwell_sec, dwell_bucket(ev.dwell_sec),
        )
        return ev

    def finalize(self, media_ts: float) -> list[DwellEvent]:
        """Flush anyone still inside a zone when the run ends (e.g. EOF)."""
        events: list[DwellEvent] = []
        for zd in self._zones:
            for tid, st in list(zd._tracks.items()):
                st.last_seen_at = max(st.last_seen_at, media_ts)
                ev = self._finalize_one(zd, tid, st)
                if ev is not None:
                    events.append(ev)
        return events

    def summary(self) -> list[dict]:
        out = []
        for zd in self._zones:
            sizes = [e.dwell_sec for e in zd._events]
            histogram: dict[str, int] = {}
            for s in sizes:
                b = dwell_bucket(s)
                histogram[b] = histogram.get(b, 0) + 1
            out.append({
                "zone_id": zd.zone_id,
                "zone_name": zd.zone_name,
                "visits": len(sizes),
                "currently_inside": len(zd._tracks),
                "avg_dwell_sec": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
                "median_dwell_sec": round(median(sizes), 1) if sizes else 0.0,
                "max_dwell_sec": round(max(sizes), 1) if sizes else 0.0,
                "bucket_histogram": {
                    label: histogram[label] for _, _, label in DWELL_BUCKETS if label in histogram
                },
            })
        return out
