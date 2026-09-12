"""Per-camera inference loop.

Per frame:
  1. Pull frame
  2. YOLO detect / ByteTrack
  3. Optional pose on each person crop
  4. Feature hooks (footfall, queue, demographics, kit) — each flagged
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import get_settings
from app.features.queue_length import QueueMonitor
from app.footfall import EntranceCounter
from app.inference.pose_backend import PoseBackend
from app.inference.yolo_backend import YoloBackend
from app.models import ZoneConfig
from app.rtsp.puller import FramePuller

logger = logging.getLogger(__name__)


@dataclass
class CameraSpec:
    camera_id: str
    source: str
    zones: list[ZoneConfig] = field(default_factory=list)


class CameraLoop:
    def __init__(self, spec: CameraSpec, *, once: bool | None = None) -> None:
        self.spec = spec
        settings = get_settings()
        self._once = settings.worker_mode.lower() == "once" if once is None else once
        self._yolo = YoloBackend()
        self._pose: PoseBackend | None = None
        self._needs_pose = settings.feature_pose
        self._puller = FramePuller(spec.source, camera_id=spec.camera_id, once=self._once)
        self._fps = settings.inference_fps
        self._stop = False
        self._footfall = EntranceCounter(spec.zones)
        self._use_tracking = settings.feature_footfall and self._footfall.enabled
        self._queue = QueueMonitor(spec.zones)
        self._frames = 0
        self._started_at = 0.0

    def _get_pose(self) -> PoseBackend:
        if self._pose is None:
            self._pose = PoseBackend()
        return self._pose

    def run(self) -> None:
        logger.info(
            "camera_loop start camera=%s once=%s tracking=%s",
            self.spec.camera_id,
            self._once,
            self._use_tracking,
        )
        self._started_at = time.time()
        self._frames = 0
        try:
            for frame, wall_ts, media_ts in self._puller.frames():
                if self._stop:
                    break
                self._process_frame(frame, wall_ts, media_ts)
                self._frames += 1
        except Exception as exc:
            logger.exception("camera_loop crashed camera=%s: %s", self.spec.camera_id, exc)
        finally:
            elapsed = time.time() - self._started_at if self._started_at else 0
            logger.info(
                "camera_loop done camera=%s frames=%d elapsed=%.1fs footfall=%s",
                self.spec.camera_id,
                self._frames,
                elapsed,
                self._footfall.summary(),
            )

    def _process_frame(self, frame: np.ndarray, wall_ts: float, media_ts: float) -> None:
        settings = get_settings()
        h, w = frame.shape[:2]

        if self._use_tracking:
            detections = self._yolo.detect_tracked(frame)
        else:
            detections = self._yolo.detect(frame)

        poses: dict[int, list] = {}
        if self._needs_pose:
            pose = self._get_pose()
            for i, det in enumerate(detections):
                if det.class_name != "person":
                    continue
                x1, y1, x2, y2 = (
                    max(0, int(det.x1)), max(0, int(det.y1)),
                    min(w, int(det.x2)), min(h, int(det.y2)),
                )
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                kps = pose.estimate(crop)
                if kps:
                    poses[i] = kps

        if settings.feature_footfall:
            crossings = self._footfall.update(detections, w, h, media_ts=media_ts)
            if crossings and settings.feature_demographics:
                from app.features.demographics import estimate as estimate_demographics

                for ev in crossings:
                    demo = estimate_demographics(frame, ev)
                    logger.info(
                        "demographics track=%s gender=%s age_group=%s age=%s",
                        ev.track_id,
                        demo.gender if demo else None,
                        demo.age_group if demo else None,
                        demo.age_estimate if demo else None,
                    )

        if settings.feature_queue and self._queue.enabled:
            queues = self._queue.estimate(detections, w, h)
            logger.info("queue %s", [q.__dict__ for q in queues])

        if settings.feature_kit:
            from app.features.kit_detection import estimate as estimate_kit

            hits = estimate_kit(detections, self.spec.zones, w, h)
            if hits:
                logger.info(
                    "kit hits=%s",
                    [(h.class_name, round(h.confidence, 2), h.zone_id) for h in hits],
                )

        if poses:
            logger.debug("pose on %d person crops", len(poses))
