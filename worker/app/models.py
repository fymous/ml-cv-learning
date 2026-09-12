"""Shared dataclasses used by the loop, zones, and feature modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ZoneConfig:
    zone_id: str
    zone_type: str
    polygon: list[list[float]] = field(default_factory=list)
    name: str = ""
    # Optional directed tripwire for entrance counting: [[ax, ay], [bx, by]].
    # When present on an `entrance` zone, footfall counts directional line
    # crossings instead of polygon presence (kills the enter-then-exit=2 bug).
    line: list[list[float]] | None = None
    # Optional vector pointing "into" the venue, e.g. [0, -1] for "up".
    # If omitted, the inside is the left-hand side of the a->b arrow.
    in_direction: list[float] | None = None
