"""Kit / object presence in a zone — stub to implement on the lab laptop.

Idea: given YOLO detections + a `kit` zone, report which objects of interest
are inside (backpack, suitcase, handbag, bottle, laptop, …).

This file is the template for a new capability. Replace the class list and
add a better model if COCO classes are not enough (custom YOLO, PPE, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.inference.backend import Detection
from app.models import ZoneConfig
from app.zones import bbox_overlaps_zone, polygon_to_frame_space

# COCO names that often mean "someone brought a kit / bag / tool"
DEFAULT_KIT_CLASSES = frozenset(
    {
        "backpack",
        "handbag",
        "suitcase",
        "umbrella",
        "bottle",
        "cup",
        "laptop",
        "cell phone",
        "tie",
        "sports ball",
    }
)


@dataclass
class KitHit:
    zone_id: str
    zone_name: str
    class_name: str
    confidence: float
    track_id: int | None
    x1: float
    y1: float
    x2: float
    y2: float


def estimate(
    detections: list[Detection],
    zones: list[ZoneConfig],
    frame_w: int,
    frame_h: int,
    class_names: frozenset[str] = DEFAULT_KIT_CLASSES,
) -> list[KitHit]:
    hits: list[KitHit] = []
    kit_zones = [z for z in zones if z.zone_type in ("kit", "restricted", "queue")]
    if not kit_zones:
        kit_zones = list(zones)
    for z in kit_zones:
        poly = polygon_to_frame_space(z.polygon, frame_w, frame_h)
        for d in detections:
            if d.class_name not in class_names:
                continue
            if not bbox_overlaps_zone(d.x1, d.y1, d.x2, d.y2, poly):
                continue
            hits.append(
                KitHit(
                    zone_id=z.zone_id,
                    zone_name=z.name or z.zone_type,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    track_id=d.track_id,
                    x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                )
            )
    return hits
