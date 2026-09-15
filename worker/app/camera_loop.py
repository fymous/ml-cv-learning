"""Per-camera inference loop.

Per frame:
  1. Pull frame
  2. YOLO detect / ByteTrack
  3. Optional pose on each person crop
  4. Feature hooks (footfall, group, staff filter, demographics, dwell,
     queue, kit) — each independently flagged
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import get_settings
from app.features.attendance import AttendanceMonitor
from app.features.dwell_time import DwellTracker
from app.features.group_count import GroupCounter
from app.features.queue_length import QueueMonitor
from app.features.staff_filter import StaffFilter
from app.features.staff_identity import StaffProfile
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
    # Optional duck-typed sink for external tools (e.g. the local UI) to capture
    # per-event screenshots. Must expose:
    #   on_crossing(frame, detections, exclude_ids, ev, demo, group_result,
    #               footfall_summary, media_ts)
    # May optionally also expose (each checked with hasattr — older sinks
    # without them simply don't get those callbacks):
    #   on_dwell(frame, detections, exclude_ids, dwell_event, media_ts)
    #   on_attendance(frame, detections, exclude_ids, attendance_event, media_ts)
    # Kept generic so the worker never imports UI code.
    event_sink: object | None = None


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
        self._group = GroupCounter() if settings.feature_group else None
        # Exclude guards/staff/passers-by from the count (staff/exclude zones +
        # optional uniform colour). Only active if such a zone or colour is set.
        self._staff = (
            StaffFilter(spec.zones, self._staff_uniform_ranges(settings))
            if settings.feature_staff_filter
            else None
        )
        # Demographics aggregator (multi-frame voting). Module import is light;
        # the heavy InsightFace model loads lazily on first face read.
        self._demog = None
        self._demog_tally = None
        if settings.feature_demographics:
            from app.features.demographics import (
                DemographicsAggregator,
                DemographicsTally,
            )

            self._demog = DemographicsAggregator()
            self._demog_tally = DemographicsTally()
        # Time-in-zone browsing dwell — independent of footfall, keys off its
        # own `dwell`-type zones. No-op if none are drawn.
        self._dwell = (
            DwellTracker(spec.zones, min_dwell_sec=settings.dwell_min_sec)
            if settings.feature_dwell
            else None
        )
        # Customer attendance / unattended alerting — runs in `service` zones,
        # relies on the staff filter to identify staff. No-op without a zone.
        self._attend = (
            AttendanceMonitor(
                spec.zones,
                proximity_m=settings.attend_proximity_m,
                attend_min_sec=settings.attend_min_sec,
                unattended_alert_sec=settings.unattended_alert_sec,
            )
            if settings.feature_attendance
            else None
        )
        self._last_media_ts = 0.0
        # Tracking is needed by footfall, group, and demographics (all key off
        # stable track IDs and only make sense with an entrance zone present),
        # or independently by dwell (keys off its own dwell zones).
        self._use_tracking = (
            self._footfall.enabled
            and (
                settings.feature_footfall
                or settings.feature_group
                or settings.feature_demographics
            )
        ) or (self._dwell is not None and self._dwell.enabled) \
            or (self._attend is not None and self._attend.enabled)
        self._queue = QueueMonitor(spec.zones)
        self._frames = 0
        self._started_at = 0.0

    @staticmethod
    def _staff_uniform_ranges(settings) -> list[list[int]]:
        """Uniform HSV ranges for the staff filter: hand-configured ranges plus
        any learned during onboarding (staff_profile.json)."""
        ranges = [list(r) for r in settings.staff_uniform_hsv]
        if settings.staff_profile_path:
            profile = StaffProfile.load(settings.staff_profile_path)
            if profile and profile.enabled:
                ranges.extend(profile.uniform_hsv)
                logger.info(
                    "loaded staff profile %s (%d uniform ranges, %d samples)",
                    settings.staff_profile_path, len(profile.uniform_hsv), profile.sample_count,
                )
        return ranges

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
            if self._group is not None:
                logger.info("group summary %s", self._group.summary())
            if self._staff is not None and self._staff.enabled:
                logger.info("staff summary %s", self._staff.summary())
            if self._demog_tally is not None:
                logger.info("demographics summary %s", self._demog_tally.summary())
            if self._dwell is not None and self._dwell.enabled:
                self._dwell.finalize(self._last_media_ts)
                logger.info("dwell summary %s", self._dwell.summary())
            if self._attend is not None and self._attend.enabled:
                logger.info("attendance summary %s", self._attend.summary())

    def _process_frame(self, frame: np.ndarray, wall_ts: float, media_ts: float) -> None:
        settings = get_settings()
        h, w = frame.shape[:2]
        self._last_media_ts = media_ts

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

        # Tag staff/guards/passers-by to exclude from the count.
        exclude_ids: set[int] = set()
        if self._staff is not None and self._staff.enabled:
            exclude_ids = self._staff.mark(detections, w, h, frame)

        # Browsing dwell time — its own zones, staff excluded (restocking a
        # shelf isn't a customer visit).
        if self._dwell is not None and self._dwell.enabled:
            dwell_events = self._dwell.update(detections, w, h, media_ts, exclude_ids)
            sink = self.spec.event_sink
            if dwell_events and sink is not None and hasattr(sink, "on_dwell"):
                for dev in dwell_events:
                    try:
                        sink.on_dwell(frame, detections, exclude_ids, dev, media_ts)
                    except Exception:  # noqa: BLE001 — a capture bug must not kill the run
                        logger.exception("event_sink.on_dwell failed")

        # Customer attendance / unattended alerting. `exclude_ids` are the
        # staff-tagged tracks — everyone else in a service zone is a customer.
        if self._attend is not None and self._attend.enabled:
            attend_events = self._attend.update(
                detections, w, h, media_ts, staff_ids=exclude_ids
            )
            sink = self.spec.event_sink
            if attend_events and sink is not None and hasattr(sink, "on_attendance"):
                for aev in attend_events:
                    try:
                        sink.on_attendance(frame, detections, exclude_ids, aev, media_ts)
                    except Exception:  # noqa: BLE001 — a capture bug must not kill the run
                        logger.exception("event_sink.on_attendance failed")

        # Demographics: accumulate per-track face votes across frames (only the
        # gated, good-quality faces are kept). Resolved once, on the enter event.
        # Skip excluded (staff) tracks — no point reading a guard's face.
        if self._demog is not None:
            active_ids: set[int] = set()
            for det in detections:
                if (
                    det.track_id is not None
                    and det.class_name == "person"
                    and det.track_id not in exclude_ids
                ):
                    active_ids.add(det.track_id)
                    self._demog.observe(frame, det)
            self._demog.prune(active_ids)

        # Footfall runs if either footfall or group is on (group needs the
        # directional enter events footfall produces).
        crossings: list = []
        run_crossings = self._footfall.enabled and (
            settings.feature_footfall or settings.feature_group
        )
        if run_crossings:
            crossings = self._footfall.update(
                detections, w, h, media_ts=media_ts, exclude_ids=exclude_ids
            )
            sink = self.spec.event_sink
            for ev in crossings:
                demo = None
                group_result = None
                if ev.direction == "enter":
                    if self._demog is not None:
                        demo = self._demog.result(ev.track_id)
                        if self._demog_tally is not None:
                            self._demog_tally.add(demo)
                        logger.info(
                            "demographics track=%s gender=%s age_group=%s age=%s votes=%s",
                            ev.track_id,
                            demo.gender,
                            demo.age_group,
                            round(demo.age_estimate, 1) if demo.age_estimate is not None else None,
                            demo.votes,
                        )
                    if self._group is not None:
                        group_result = self._group.add(ev, media_ts, w, h)
                if sink is not None:
                    try:
                        sink.on_crossing(
                            frame, detections, exclude_ids, ev, demo,
                            group_result, self._footfall.summary(), media_ts,
                        )
                    except Exception:  # noqa: BLE001 — a capture bug must not kill the run
                        logger.exception("event_sink.on_crossing failed")

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
