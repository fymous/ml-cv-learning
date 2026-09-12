"""Zone polygon helpers — point-in-polygon and bbox-in-zone checks."""

from __future__ import annotations

import numpy as np


def polygon_to_frame_space(
    polygon: list[list[float]], frame_w: int, frame_h: int
) -> list[list[float]]:
    """Map stored zone polygon into the current frame's pixel space.

    Supports:
      - normalized 0–1 coords (preferred)
      - already-full-frame pixel coords
    """
    if not polygon or frame_w <= 0 or frame_h <= 0:
        return polygon

    max_x = max(float(p[0]) for p in polygon)
    max_y = max(float(p[1]) for p in polygon)

    if max_x <= 1.5 and max_y <= 1.5:
        return [[float(p[0]) * frame_w, float(p[1]) * frame_h] for p in polygon]

    return [[float(p[0]), float(p[1])] for p in polygon]


def point_in_polygon(px: float, py: float, polygon: list[list[float]]) -> bool:
    """Ray-casting algorithm. polygon = [[x,y], [x,y], ...]"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def bbox_overlaps_zone(
    x1: float, y1: float, x2: float, y2: float, polygon: list[list[float]]
) -> bool:
    """True if the bbox centre is inside the polygon."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return point_in_polygon(cx, cy, polygon)


def iou_with_zone(
    x1: float, y1: float, x2: float, y2: float, polygon: list[list[float]]
) -> float:
    """Fraction of the bbox that falls inside the polygon (approx via sampling)."""
    pts_x = np.linspace(x1, x2, 10)
    pts_y = np.linspace(y1, y2, 10)
    total = 0
    inside = 0
    for px in pts_x:
        for py in pts_y:
            total += 1
            if point_in_polygon(float(px), float(py), polygon):
                inside += 1
    return inside / total if total else 0.0
