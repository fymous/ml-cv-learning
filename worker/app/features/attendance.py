"""Customer attendance / unattended-customer alerting.

Question this answers: "a customer is browsing in the service area — did a
*staff member* attend to them, and if not, for how long have they been left
waiting?" It raises an alert when a customer (or a group of customers) has
been present in a `service`-type zone for `unattended_alert_sec` with no staff
coming close, and it LATCHES `attended` the moment a staff member does spend
`attend_min_sec` near them — so a customer who was helped once is never
re-alerted, even if they then browse alone for ages.

Design decisions (why it's built this way):

* **Attendance is anchored on positive STAFF identification, never on
  "someone is nearby".** Staff ids come from `features/staff_filter.py`
  (uniform colour + staff zone, sticky per track). A customer's *friend* is a
  customer — never staff-tagged — so a friend standing right next to them can
  never be mistaken for "being attended". This is the whole reason we key off
  staff ids rather than generic proximity.

* **Groups are handled at the party level.** Customers standing close together
  form a party (per-frame proximity clustering). If staff attends *any* member
  the whole party is marked attended, and only ONE unattended alert is raised
  per party — so three friends waiting together produce one alert, not three,
  and helping one of them clears all.

* **Perspective-aware proximity.** Raw pixel distance is depth-blind, so we
  normalise by bounding-box height (people are ~1.7 m tall) to approximate
  real-world metres without needing a homography. Good enough for "is staff
  within ~1.5 m"; swap in a ground-plane homography later for precision.

No extra model, no per-frame inference — pure geometry + a small state machine
on detections the pipeline already has. Requires the staff filter to be
enabled (otherwise no track is ever staff and every customer eventually
alerts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import hypot

from app.inference.backend import Detection
from app.models import ZoneConfig
from app.zones import point_in_polygon, polygon_to_frame_space

logger = logging.getLogger(__name__)

_ZONE_TYPE = "service"

# Assumed average standing person height, for pixel->metre scale.
_PERSON_H_M = 1.7
# Two customers within this many metres are treated as the same party.
_PARTY_RADIUS_M = 1.2
# Cap the per-frame dt used to accumulate attention (guards against long gaps
# between sampled frames inflating the timer).
_MAX_DT = 1.0
# Forget a customer not seen for this many update cycles.
_MISS_GRACE_FRAMES = 15
_INF = float("inf")


def _feet_point(d: Detection) -> tuple[float, float]:
    cx = (d.x1 + d.x2) / 2
    h = max(1.0, d.y2 - d.y1)
    cy = d.y2 - 0.08 * h
    return cx, cy


def _dist_m(a_feet: tuple[float, float], a_h: float,
            b_feet: tuple[float, float], b_h: float) -> float:
    """Approximate ground distance in metres between two people, using the
    average bbox height as a per-depth pixels-per-metre scale."""
    px = hypot(a_feet[0] - b_feet[0], a_feet[1] - b_feet[1])
    avg_h = max(1.0, (a_h + b_h) / 2.0)
    px_per_m = avg_h / _PERSON_H_M
    return px / px_per_m


@dataclass
class AttendanceEvent:
    kind: str            # "attended" | "unattended"
    zone_id: str
    zone_name: str
    track_id: int        # the customer (representative, for a party)
    wait_sec: float      # seconds waited before being attended / before alert
    x1: float
    y1: float
    x2: float
    y2: float
    staff_track_id: int | None = None
    party_size: int = 1


@dataclass
class _Cust:
    first_seen: float
    last_seen: float
    x1: float
    y1: float
    x2: float
    y2: float
    fx: float
    fy: float
    h: float
    zone_id: str
    attention_accum: float = 0.0
    attended: bool = False
    alerted: bool = False
    missed: int = 0


@dataclass
class _ServiceZone:
    zone_id: str
    zone_name: str
    polygon: list[list[float]]
    customers_seen: set[int] = field(default_factory=set)
    attended: set[int] = field(default_factory=set)
    alerted: set[int] = field(default_factory=set)


class AttendanceMonitor:
    def __init__(
        self,
        zones: list[ZoneConfig],
        proximity_m: float = 1.5,
        attend_min_sec: float = 2.0,
        unattended_alert_sec: float = 5.0,
    ) -> None:
        self._zones = [
            _ServiceZone(zone_id=z.zone_id, zone_name=z.name or z.zone_type, polygon=z.polygon)
            for z in zones
            if z.zone_type == _ZONE_TYPE and z.polygon
        ]
        self._by_id = {z.zone_id: z for z in self._zones}
        self._proximity_m = proximity_m
        self._attend_min_sec = attend_min_sec
        self._alert_sec = unattended_alert_sec
        self._cust: dict[int, _Cust] = {}
        self._prev_ts: float | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._zones)

    def update(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        media_ts: float,
        staff_ids: set[int] | None = None,
    ) -> list[AttendanceEvent]:
        if not self._zones:
            return []
        staff_ids = staff_ids or set()
        dt = 0.0 if self._prev_ts is None else max(0.0, min(media_ts - self._prev_ts, _MAX_DT))
        self._prev_ts = media_ts

        persons = [d for d in detections if d.class_name == "person" and d.track_id is not None]
        staff_feet = [
            (_feet_point(d), max(1.0, d.y2 - d.y1)) for d in persons if d.track_id in staff_ids
        ]

        polys = {z.zone_id: polygon_to_frame_space(z.polygon, frame_w, frame_h) for z in self._zones}

        active: set[int] = set()
        now: list[tuple[int, tuple[float, float], float]] = []  # (tid, feet, h)
        for d in persons:
            tid = d.track_id
            if tid in staff_ids:
                continue
            feet = _feet_point(d)
            ph = max(1.0, d.y2 - d.y1)
            zid = next((z.zone_id for z in self._zones
                        if point_in_polygon(feet[0], feet[1], polys[z.zone_id])), None)
            if zid is None:
                continue
            active.add(tid)
            st = self._cust.get(tid)
            if st is None:
                st = _Cust(first_seen=media_ts, last_seen=media_ts, x1=d.x1, y1=d.y1,
                           x2=d.x2, y2=d.y2, fx=feet[0], fy=feet[1], h=ph, zone_id=zid)
                self._cust[tid] = st
            else:
                st.last_seen = media_ts
                st.missed = 0
                st.x1, st.y1, st.x2, st.y2 = d.x1, d.y1, d.x2, d.y2
                st.fx, st.fy, st.h, st.zone_id = feet[0], feet[1], ph, zid
            self._by_id[zid].customers_seen.add(tid)
            now.append((tid, feet, ph))

        events: list[AttendanceEvent] = []

        # 1) staff proximity → accumulate attention → latch attended (direct).
        near_now: set[int] = set()
        for tid, feet, ph in now:
            st = self._cust[tid]
            best, best_sid = _INF, None
            for sfeet, sh in staff_feet:
                dm = _dist_m(feet, ph, sfeet, sh)
                if dm < best:
                    best = dm
            if best <= self._proximity_m:
                near_now.add(tid)
                if not st.attended:
                    st.attention_accum += dt
                    if st.attention_accum >= self._attend_min_sec:
                        st.attended = True
                        self._by_id[st.zone_id].attended.add(tid)
                        events.append(AttendanceEvent(
                            "attended", st.zone_id, self._by_id[st.zone_id].zone_name, tid,
                            round(media_ts - st.first_seen, 1), st.x1, st.y1, st.x2, st.y2,
                        ))
                        logger.info(
                            "attendance ATTENDED zone=%s customer=%s waited=%.1fs t=%.1f",
                            st.zone_id, tid, media_ts - st.first_seen, media_ts,
                        )

        # 2) parties via current-proximity components.
        comps = self._components(now)

        # 3) propagate attended across a party (staff helped one → all helped).
        for comp in comps:
            if any(self._cust[t].attended for t in comp):
                for t in comp:
                    if not self._cust[t].attended:
                        self._cust[t].attended = True
                        self._by_id[self._cust[t].zone_id].attended.add(t)

        # 4) unattended alert — one per party, once, and never if attended or
        #    if staff is currently approaching a member.
        for comp in comps:
            if any(self._cust[t].attended for t in comp):
                continue
            if any(self._cust[t].alerted for t in comp):
                continue
            if any(t in near_now for t in comp):
                continue
            wait = media_ts - min(self._cust[t].first_seen for t in comp)
            if wait >= self._alert_sec:
                rep = min(comp, key=lambda t: self._cust[t].first_seen)
                st = self._cust[rep]
                for t in comp:
                    self._cust[t].alerted = True
                self._by_id[st.zone_id].alerted.add(rep)
                events.append(AttendanceEvent(
                    "unattended", st.zone_id, self._by_id[st.zone_id].zone_name, rep,
                    round(wait, 1), st.x1, st.y1, st.x2, st.y2, party_size=len(comp),
                ))
                logger.info(
                    "attendance UNATTENDED zone=%s customer=%s party=%d waited=%.1fs t=%.1f",
                    st.zone_id, rep, len(comp), wait, media_ts,
                )

        # 5) age out.
        for tid in list(self._cust.keys()):
            if tid in active:
                continue
            self._cust[tid].missed += 1
            if self._cust[tid].missed > _MISS_GRACE_FRAMES:
                del self._cust[tid]

        return events

    def _components(self, now: list[tuple[int, tuple[float, float], float]]) -> list[set[int]]:
        """Union-find over currently-visible customers by standing proximity."""
        parent = {tid: tid for tid, _, _ in now}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for i in range(len(now)):
            ti, fi, hi = now[i]
            for j in range(i + 1, len(now)):
                tj, fj, hj = now[j]
                if _dist_m(fi, hi, fj, hj) <= _PARTY_RADIUS_M:
                    union(ti, tj)
        comps: dict[int, set[int]] = {}
        for tid, _, _ in now:
            comps.setdefault(find(tid), set()).add(tid)
        return list(comps.values())

    def summary(self) -> list[dict]:
        out = []
        for z in self._zones:
            waiting = sum(
                1 for st in self._cust.values()
                if st.zone_id == z.zone_id and not st.attended and st.missed == 0
            )
            out.append({
                "zone_id": z.zone_id,
                "zone_name": z.zone_name,
                "customers_seen": len(z.customers_seen),
                "attended": len(z.attended),
                "unattended_alerts": len(z.alerted),
                "currently_waiting": waiting,
            })
        return out
