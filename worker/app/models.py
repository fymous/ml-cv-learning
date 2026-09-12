"""Shared dataclasses used by the loop, zones, and feature modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZoneConfig:
    zone_id: str
    zone_type: str
    polygon: list[list[float]]
    name: str = ""
