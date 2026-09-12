from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inference_fps: float = 2.0
    yolo_model: str = "yolo11n.pt"
    yolo_conf_threshold: float = 0.4
    pose_conf_threshold: float = 0.5

    # Tracker config passed to Ultralytics .track(). botsort.yaml is ReID/appearance
    # aware and keeps IDs stable across brief occlusion far better than bytetrack.yaml.
    tracker: str = "botsort.yaml"

    # InsightFace model pack for demographics. buffalo_l is more accurate than
    # buffalo_s at a modest CPU cost; only runs when FEATURE_DEMOGRAPHICS is on.
    demographics_model: str = "buffalo_l"

    # once = process a file to EOF then exit; live = reconnect / loop
    worker_mode: str = "once"
    worker_once_max_sec: float = 0.0

    rtsp_reconnect_delay: int = 5
    rtsp_max_retries: int = 0  # 0 = infinite (ignored in once mode)

    # Independent feature flags — each capability can be toggled alone
    feature_footfall: bool = True
    feature_queue: bool = True
    feature_demographics: bool = False
    feature_kit: bool = False
    feature_pose: bool = False
    feature_group: bool = True

    # Exclude guards / staff / passers-by from the people count. Uses `staff` or
    # `exclude` zones, plus optional uniform-colour matching.
    feature_staff_filter: bool = True
    # Optional list of OpenCV HSV ranges [[h_lo,s_lo,v_lo,h_hi,s_hi,v_hi], ...]
    # for staff uniforms. Empty = colour matching off (zones only). Set via env
    # as JSON, e.g. STAFF_UNIFORM_HSV='[[90,60,40,130,255,255]]'.
    staff_uniform_hsv: list[list[int]] = []


@lru_cache
def get_settings() -> WorkerSettings:
    return WorkerSettings()
