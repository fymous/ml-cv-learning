"""Optional gender / age estimate with multi-frame voting.

Off by default (FEATURE_DEMOGRAPHICS). Never changes the entered count —
only annotates a CrossingEvent that EntranceCounter already decided happened.

Why voting: at a doorway the camera often sees the back or top of a head, so a
single-shot read at the enter moment frequently finds no usable face. Instead
we `observe()` a track across the frames it is visible, keep only good-quality
faces (a quality gate), and on the enter event return a **majority-vote gender
+ median age**. This is far more robust than one crop.

Fails soft: if the model cannot load or no face is found, returns unknown
fields instead of raising. CPU-only. Loaded lazily.

Model: InsightFace (SCRFD + genderage head). Pack from DEMOGRAPHICS_MODEL
(default buffalo_l — more accurate than buffalo_s at a modest CPU cost).
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from app.config import get_settings
from app.footfall import CrossingEvent
from app.inference.backend import Detection

logger = logging.getLogger(__name__)

# Age bands for age-group analysis. (low_inclusive, high_inclusive, label)
AGE_BUCKETS: list[tuple[int, int, str]] = [
    (0, 12, "0-12"),
    (13, 19, "13-19"),
    (20, 34, "20-34"),
    (35, 54, "35-54"),
    (55, 200, "55+"),
]


def age_bucket(age: float | None) -> str | None:
    if age is None:
        return None
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return None

# Quality gate — only trust a face detection that is confidently a frontal-ish,
# reasonably large face. Back-of-head / tiny / low-score crops are discarded.
_MIN_DET_SCORE = 0.60
_MIN_FACE_PX = 24
# Stop running the model on a track once we have this many good votes.
_MAX_VOTES = 7
# Drop accumulated votes for tracks not seen for this many observe() cycles.
_PRUNE_MISS = 30

_lock = threading.Lock()
_analyzer = None
_load_failed = False


@dataclass
class Demographics:
    gender: str | None  # "male" | "female" | None
    age_estimate: float | None
    age_group: str | None  # AGE_BUCKETS label, e.g. "20-34" | None
    votes: int = 0


@dataclass
class _Votes:
    genders: list[str] = field(default_factory=list)
    ages: list[float] = field(default_factory=list)
    missed: int = 0


def _get_analyzer():
    global _analyzer, _load_failed
    if _analyzer is not None or _load_failed:
        return _analyzer
    with _lock:
        if _analyzer is not None or _load_failed:
            return _analyzer
        try:
            from insightface.app import FaceAnalysis

            pack = get_settings().demographics_model
            app = FaceAnalysis(
                name=pack,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "genderage"],
            )
            app.prepare(ctx_id=-1, det_size=(320, 320))
            _analyzer = app
            logger.info("demographics: %s loaded (CPU)", pack)
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


def _best_face(crop: np.ndarray):
    """Return the largest good-quality face in the crop, or None."""
    analyzer = _get_analyzer()
    if analyzer is None or crop is None or crop.size == 0:
        return None
    try:
        faces = analyzer.get(crop)
    except Exception as exc:
        logger.debug("demographics: inference failed: %s", exc)
        return None
    good = []
    for f in faces:
        score = float(getattr(f, "det_score", 0.0))
        fw = float(f.bbox[2] - f.bbox[0])
        fh = float(f.bbox[3] - f.bbox[1])
        if score >= _MIN_DET_SCORE and min(fw, fh) >= _MIN_FACE_PX:
            good.append(f)
    if not good:
        return None
    return max(good, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _aggregate(votes: _Votes) -> Demographics:
    if not votes.genders and not votes.ages:
        return Demographics(gender=None, age_estimate=None, age_group=None, votes=0)
    gender = None
    if votes.genders:
        gender = max(set(votes.genders), key=votes.genders.count)
    age = median(votes.ages) if votes.ages else None
    n = max(len(votes.genders), len(votes.ages))
    return Demographics(gender=gender, age_estimate=age, age_group=age_bucket(age), votes=n)


class DemographicsAggregator:
    """Accumulates per-track face votes across frames, resolves on enter."""

    def __init__(self) -> None:
        self._votes: dict[int, _Votes] = defaultdict(_Votes)

    def observe(self, frame: np.ndarray, det: Detection) -> None:
        """Run a gated face read for one tracked person on this frame."""
        tid = det.track_id
        if tid is None or det.class_name != "person":
            return
        v = self._votes[tid]
        if len(v.genders) >= _MAX_VOTES:
            return
        crop = _padded_crop(frame, det.x1, det.y1, det.x2, det.y2)
        face = _best_face(crop)
        if face is None:
            return
        v.genders.append("male" if int(face.gender) == 1 else "female")
        v.ages.append(float(face.age))

    def result(self, track_id: int) -> Demographics:
        """Majority/median over the votes collected for this track so far."""
        return _aggregate(self._votes.get(track_id, _Votes()))

    def prune(self, active_ids: set[int]) -> None:
        """Forget tracks that have not been seen for a while (bound memory)."""
        for tid in list(self._votes.keys()):
            if tid in active_ids:
                self._votes[tid].missed = 0
                continue
            self._votes[tid].missed += 1
            if self._votes[tid].missed > _PRUNE_MISS:
                del self._votes[tid]


class DemographicsTally:
    """Run-level age-group / gender distribution over counted visitors."""

    def __init__(self) -> None:
        self._age = Counter()
        self._gender = Counter()
        self._age_gender = Counter()
        self._with_reading = 0
        self._total = 0

    def add(self, demo: Demographics | None) -> None:
        self._total += 1
        if demo is None:
            return
        ag = demo.age_group or "unknown"
        g = demo.gender or "unknown"
        if demo.age_group or demo.gender:
            self._with_reading += 1
        self._age[ag] += 1
        self._gender[g] += 1
        self._age_gender[f"{ag}/{g}"] += 1

    def summary(self) -> dict:
        # keep age groups in a stable, meaningful order
        order = [b[2] for b in AGE_BUCKETS] + ["unknown"]
        by_age = {k: self._age[k] for k in order if self._age.get(k)}
        return {
            "counted": self._total,
            "with_reading": self._with_reading,
            "by_age_group": by_age,
            "by_gender": dict(self._gender),
            "by_age_gender": dict(sorted(self._age_gender.items())),
        }


def estimate(frame: np.ndarray, event: CrossingEvent) -> Demographics | None:
    """Single-shot read on the enter crop (backward-compatible entrypoint)."""
    if _get_analyzer() is None:
        return None
    crop = _padded_crop(frame, event.x1, event.y1, event.x2, event.y2)
    face = _best_face(crop)
    if face is None:
        return Demographics(gender=None, age_estimate=None, age_group=None, votes=0)
    gender = "male" if int(face.gender) == 1 else "female"
    age = float(face.age)
    return Demographics(gender=gender, age_estimate=age, age_group=age_bucket(age), votes=1)
