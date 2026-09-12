"""Optional per-entry gender / age estimate.

Off by default (FEATURE_DEMOGRAPHICS). Never changes the entered count —
only annotates a CrossingEvent that EntranceCounter already decided happened.

Fails soft: if the model cannot load or no face is found, returns unknown
fields instead of raising. CPU-only. Loaded lazily.

Model: InsightFace buffalo_s (SCRFD-500MF + genderage head, ~160MB).
Runs on the person crop only — one inference per counted enter, not per frame.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from app.footfall import CrossingEvent

logger = logging.getLogger(__name__)

_KID_AGE_MAX = 13
_MODEL_PACK = "buffalo_s"

_lock = threading.Lock()
_analyzer = None
_load_failed = False


@dataclass
class Demographics:
    gender: str | None  # "male" | "female" | None
    age_estimate: float | None
    age_group: str | None  # "kid" | "adult" | None


def _get_analyzer():
    global _analyzer, _load_failed
    if _analyzer is not None or _load_failed:
        return _analyzer
    with _lock:
        if _analyzer is not None or _load_failed:
            return _analyzer
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=_MODEL_PACK,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "genderage"],
            )
            app.prepare(ctx_id=-1, det_size=(320, 320))
            _analyzer = app
            logger.info("demographics: %s loaded (CPU)", _MODEL_PACK)
        except Exception as exc:
            _load_failed = True
            logger.warning("demographics: model unavailable — %s", exc)
    return _analyzer


def _padded_crop(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float, pad: float = 0.35
) -> np.ndarray | None:
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px1 = max(0, int(x1 - bw * pad))
    py1 = max(0, int(y1 - bh * pad))
    px2 = min(w, int(x2 + bw * pad))
    py2 = min(h, int(y2 + bh * pad))
    if px2 <= px1 or py2 <= py1:
        return None
    return frame[py1:py2, px1:px2]


def estimate(frame: np.ndarray, event: CrossingEvent) -> Demographics | None:
    analyzer = _get_analyzer()
    if analyzer is None:
        return None

    crop = _padded_crop(frame, event.x1, event.y1, event.x2, event.y2)
    if crop is None or crop.size == 0:
        return Demographics(gender=None, age_estimate=None, age_group=None)

    try:
        faces = analyzer.get(crop)
    except Exception as exc:
        logger.debug("demographics: inference failed track=%s: %s", event.track_id, exc)
        return Demographics(gender=None, age_estimate=None, age_group=None)

    if not faces:
        return Demographics(gender=None, age_estimate=None, age_group=None)

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    gender = "male" if int(face.gender) == 1 else "female"
    age = float(face.age)
    age_group = "kid" if age < _KID_AGE_MAX else "adult"
    return Demographics(gender=gender, age_estimate=age, age_group=age_group)
