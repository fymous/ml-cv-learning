"""Run the worker pipeline once for the UI, capturing per-event screenshots.

Reuses the real `CameraLoop` (no duplicated CV logic). Sets feature flags via
env, attaches an EventCapture sink, runs to EOF, then reads the final summaries
straight off the loop's feature objects.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the worker package importable.
_WORKER = Path(__file__).resolve().parents[1] / "worker"
if str(_WORKER) not in sys.path:
    sys.path.insert(0, str(_WORKER))

# Persistence-tuned tracker for busy doorways (fewer ID switches).
_PERSIST_TRACKER = str(_WORKER / "app" / "trackers" / "botsort_persist.yaml")


def run_pipeline(
    video_path: str,
    zones_yaml: str,
    out_dir: str,
    *,
    fps: float = 5.0,
    footfall: bool = True,
    group: bool = True,
    demographics: bool = True,
    staff_filter: bool = True,
    dwell: bool = True,
    dwell_min_sec: float = 4.0,
    attendance: bool = True,
    unattended_alert_sec: float = 5.0,
    staff_profile_path: str | None = None,
    tracker: str | None = None,
    yolo_model: str = "yolo11s.pt",
) -> dict:
    tracker = tracker or (_PERSIST_TRACKER if os.path.exists(_PERSIST_TRACKER) else "botsort.yaml")
    os.makedirs(out_dir, exist_ok=True)

    os.environ["WORKER_MODE"] = "once"
    os.environ["INFERENCE_FPS"] = str(fps)
    os.environ["TRACKER"] = tracker
    os.environ["YOLO_MODEL"] = yolo_model
    os.environ["FEATURE_FOOTFALL"] = str(footfall).lower()
    os.environ["FEATURE_GROUP"] = str(group).lower()
    os.environ["FEATURE_DEMOGRAPHICS"] = str(demographics).lower()
    os.environ["FEATURE_STAFF_FILTER"] = str(staff_filter).lower()
    os.environ["FEATURE_DWELL"] = str(dwell).lower()
    os.environ["DWELL_MIN_SEC"] = str(dwell_min_sec)
    os.environ["FEATURE_ATTENDANCE"] = str(attendance).lower()
    os.environ["UNATTENDED_ALERT_SEC"] = str(unattended_alert_sec)
    if staff_profile_path and os.path.exists(staff_profile_path):
        os.environ["STAFF_PROFILE_PATH"] = staff_profile_path
    else:
        os.environ.pop("STAFF_PROFILE_PATH", None)
    os.environ["FEATURE_QUEUE"] = "false"
    os.environ["FEATURE_KIT"] = "false"
    os.environ["FEATURE_POSE"] = "false"

    # Import AFTER env is set and clear the settings cache so flags take effect.
    from app.camera_loop import CameraLoop, CameraSpec
    from app.config import get_settings
    from app.zone_loader import load_zones

    get_settings.cache_clear()

    from event_capture import EventCapture  # local UI module

    zones = load_zones(zones_yaml)
    sink = EventCapture(out_dir, zones=zones)
    spec = CameraSpec(
        camera_id="ui",
        source=video_path,
        zones=zones,
        event_sink=sink,
    )
    loop = CameraLoop(spec, once=True)
    loop.run()

    summary = {
        "video": video_path,
        "zones_yaml": zones_yaml,
        "frames": loop._frames,
        "footfall": loop._footfall.summary(),
        "events_jsonl": os.path.join(out_dir, "events.jsonl"),
    }
    if getattr(loop, "_group", None) is not None:
        summary["group"] = loop._group.summary()
    if getattr(loop, "_staff", None) is not None and loop._staff.enabled:
        summary["staff"] = loop._staff.summary()
    if getattr(loop, "_demog_tally", None) is not None:
        summary["demographics"] = loop._demog_tally.summary()
    if getattr(loop, "_dwell", None) is not None and loop._dwell.enabled:
        summary["dwell"] = loop._dwell.summary()
        summary["dwell_min_sec"] = dwell_min_sec
    if getattr(loop, "_attend", None) is not None and loop._attend.enabled:
        summary["attendance"] = loop._attend.summary()
        summary["unattended_alert_sec"] = unattended_alert_sec

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def learn_staff_uniform(
    video_path: str,
    zones_yaml: str,
    profile_path: str,
    *,
    fps: float = 4.0,
    seconds: float = 15.0,
    yolo_model: str = "yolo11n.pt",
    tracker: str | None = None,
) -> dict:
    """Onboarding: learn the store's staff uniform colour from people standing
    in an `enroll` zone. Runs YOLO over the first `seconds` of video, samples
    torso colour of anyone in the zone, builds a StaffProfile, saves it, and
    returns a summary plus a preview image path.
    """
    import cv2
    import numpy as np

    tracker = tracker or (_PERSIST_TRACKER if os.path.exists(_PERSIST_TRACKER) else "botsort.yaml")
    os.environ["WORKER_MODE"] = "once"
    os.environ["YOLO_MODEL"] = yolo_model
    os.environ["TRACKER"] = tracker

    from app.config import get_settings
    from app.features.staff_identity import (
        UniformLearner,
        range_mid_bgr,
        torso_hsv_median,
        uniform_spread_warning,
    )
    from app.inference.yolo_backend import YoloBackend
    from app.zone_loader import load_zones
    from app.zones import point_in_polygon, polygon_to_frame_space

    get_settings.cache_clear()

    zones = load_zones(zones_yaml)
    enroll_zones = [z for z in zones if z.zone_type == "enroll"]
    if not enroll_zones:
        raise ValueError("No `enroll` zone found — draw one in the Zone Builder first.")

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / max(0.5, fps))))
    max_frames = int(seconds * src_fps)

    backend = YoloBackend()
    learner = UniformLearner(source=os.path.basename(video_path))
    preview = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx > max_frames:
            break
        if idx % step != 0:
            idx += 1
            continue
        h, w = frame.shape[:2]
        polys = [polygon_to_frame_space(z.polygon, w, h) for z in enroll_zones if z.polygon]
        dets = backend.detect_tracked(frame)
        annotated = frame.copy()
        for z in enroll_zones:
            pts = np.array([[int(px * w), int(py * h)] for px, py in z.polygon], np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        for d in dets:
            if d.class_name != "person":
                continue
            cx = (d.x1 + d.x2) / 2
            bh = max(1.0, d.y2 - d.y1)
            fy = d.y2 - 0.08 * bh
            if not any(point_in_polygon(cx, fy, poly) for poly in polys):
                continue
            if learner.add_sample(frame, d):
                med = torso_hsv_median(frame, d)
                bgr = range_mid_bgr([med[0], med[1], med[2], med[0], med[1], med[2]]) if med is not None else (0, 200, 0)
                cv2.rectangle(annotated, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 200, 0), 2)
                cv2.rectangle(annotated, (int(d.x1), int(d.y1) - 18),
                              (int(d.x1) + 18, int(d.y1)), bgr, -1)
        preview = annotated
        idx += 1
    cap.release()

    profile = learner.build()
    profile.save(profile_path)

    preview_path = None
    if preview is not None:
        preview_path = str(Path(profile_path).with_suffix(".preview.jpg"))
        cv2.imwrite(preview_path, preview)

    return {
        "profile_path": profile_path,
        "uniform_hsv": profile.uniform_hsv,
        "sample_count": profile.sample_count,
        "tracks_sampled": profile.tracks_sampled,
        "preview": preview_path,
        "swatches": [range_mid_bgr(r) for r in profile.uniform_hsv],
        "warning": uniform_spread_warning(profile.uniform_hsv),
    }


def first_frame(video_path: str):
    """Return (frame_bgr, w, h) for the first readable frame, or (None, 0, 0)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, 0, 0
    h, w = frame.shape[:2]
    return frame, w, h
