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


def line_to_frame_space(
    line: list[list[float]], frame_w: int, frame_h: int
) -> list[list[float]]:
    """Map a stored tripwire line into the current frame's pixel space.

    Same normalized-vs-pixel detection as `polygon_to_frame_space`.
    """
    if not line or frame_w <= 0 or frame_h <= 0:
        return line
    max_x = max(float(p[0]) for p in line)
    max_y = max(float(p[1]) for p in line)
    if max_x <= 1.5 and max_y <= 1.5:
        return [[float(p[0]) * frame_w, float(p[1]) * frame_h] for p in line]
    return [[float(p[0]), float(p[1])] for p in line]


def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Signed area * 2 of triangle abc. >0 = c is left of a->b, <0 = right."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """True if segment p1p2 crosses segment p3p4 (proper intersection)."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def crossing_is_inward(
    prev: tuple[float, float],
    cur: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    in_direction: tuple[float, float] | None = None,
) -> bool:
    """Given a movement prev->cur that crosses line a->b, is it an *enter*?

    If `in_direction` is given, enter = movement points the same way as it.
    Otherwise the inside is the left-hand side of the a->b arrow.
    """
    mvx, mvy = cur[0] - prev[0], cur[1] - prev[1]
    if in_direction is not None:
        return (mvx * in_direction[0] + mvy * in_direction[1]) > 0
    # left normal of a->b
    nx, ny = -(b[1] - a[1]), (b[0] - a[0])
    return (mvx * nx + mvy * ny) > 0


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
