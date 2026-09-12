"""Load zone polygons from a YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.models import ZoneConfig


def load_zones(path: str | Path) -> list[ZoneConfig]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    zones: list[ZoneConfig] = []
    for raw in data.get("zones", []):
        raw_line = raw.get("line")
        line = (
            [[float(p[0]), float(p[1])] for p in raw_line]
            if raw_line
            else None
        )
        raw_dir = raw.get("in_direction")
        in_direction = [float(raw_dir[0]), float(raw_dir[1])] if raw_dir else None
        zones.append(
            ZoneConfig(
                zone_id=str(raw["id"]),
                zone_type=str(raw["type"]),
                polygon=[[float(p[0]), float(p[1])] for p in raw.get("polygon", [])],
                name=str(raw.get("name") or raw["type"]),
                line=line,
                in_direction=in_direction,
            )
        )
    return zones
