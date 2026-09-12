"""Group counting — collapse people who arrive together into one group.

A family of 3 walking in together should count as **1 group of size 3**, not 3
visits. This consumes the *enter* `CrossingEvent`s that `EntranceCounter`
already produces (so it inherits the fixed, directional, de-duplicated counts)
and clusters them by **arrival time + arrival position**:

  a new enter joins an open group if it crossed within `window_sec` of that
  group's last member AND its feet-point is within `radius_frac * frame_w` of
  the group's centroid; otherwise it opens a new group.

Purely geometric and stateful — no extra model, no per-frame cost. Depends on
stable track IDs, so it benefits directly from the BoT-SORT tracker swap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import hypot

from app.footfall import CrossingEvent

logger = logging.getLogger(__name__)

_WINDOW_SEC = 4.0
_RADIUS_FRAC = 0.18  # fraction of frame width


def _feet_point(ev: CrossingEvent) -> tuple[float, float]:
    cx = (ev.x1 + ev.x2) / 2
    h = max(1.0, ev.y2 - ev.y1)
    cy = ev.y2 - 0.08 * h
    return cx, cy


@dataclass
class _Group:
    group_id: int
    members: list[int] = field(default_factory=list)
    cx: float = 0.0
    cy: float = 0.0
    started_at: float = 0.0
    last_at: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class GroupResult:
    group_id: int
    size: int
    member_track_ids: list[int]
    is_new: bool


class GroupCounter:
    def __init__(
        self, window_sec: float = _WINDOW_SEC, radius_frac: float = _RADIUS_FRAC
    ) -> None:
        self._window = window_sec
        self._radius_frac = radius_frac
        self._groups: list[_Group] = []
        self._by_track: dict[int, int] = {}
        self.total_groups = 0

    def add(
        self, event: CrossingEvent, media_ts: float, frame_w: int, frame_h: int
    ) -> GroupResult | None:
        """Assign one enter event to a group. Returns the group it landed in."""
        if event.direction != "enter":
            return None
        tid = event.track_id
        if tid in self._by_track:  # already grouped this track
            gid = self._by_track[tid]
            grp = next((g for g in self._groups if g.group_id == gid), None)
            if grp is not None:
                return GroupResult(gid, grp.size, list(grp.members), is_new=False)

        fx, fy = _feet_point(event)
        radius = self._radius_frac * max(1, frame_w)

        best: _Group | None = None
        best_dist = radius
        for g in self._groups:
            if (media_ts - g.last_at) > self._window:
                continue
            dist = hypot(fx - g.cx, fy - g.cy)
            if dist <= best_dist:
                best, best_dist = g, dist

        if best is not None:
            n = best.size
            best.cx = (best.cx * n + fx) / (n + 1)
            best.cy = (best.cy * n + fy) / (n + 1)
            best.members.append(tid)
            best.last_at = media_ts
            self._by_track[tid] = best.group_id
            logger.info(
                "group JOIN group=%d size=%d track=%s t=%.1f",
                best.group_id, best.size, tid, media_ts,
            )
            return GroupResult(best.group_id, best.size, list(best.members), is_new=False)

        self.total_groups += 1
        grp = _Group(
            group_id=self.total_groups,
            members=[tid],
            cx=fx, cy=fy,
            started_at=media_ts,
            last_at=media_ts,
        )
        self._groups.append(grp)
        self._by_track[tid] = grp.group_id
        logger.info(
            "group NEW group=%d size=1 track=%s t=%.1f",
            grp.group_id, tid, media_ts,
        )
        return GroupResult(grp.group_id, 1, [tid], is_new=True)

    def summary(self) -> dict:
        sizes = [g.size for g in self._groups]
        histogram: dict[int, int] = {}
        for s in sizes:
            histogram[s] = histogram.get(s, 0) + 1
        multi = [s for s in sizes if s >= 2]
        return {
            # every distinct arrival cluster, including solo arrivals
            "total_clusters": self.total_groups,
            # genuine groups = 2+ people arriving together
            "groups_2plus": len(multi),
            "people_in_groups": sum(multi),
            "total_people": sum(sizes),
            "avg_group_size": round(sum(multi) / len(multi), 2) if multi else 0.0,
            "size_histogram": dict(sorted(histogram.items())),
        }
