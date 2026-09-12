from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inference_fps: float = 1.0
    yolo_model: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.4
    pose_conf_threshold: float = 0.5

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


@lru_cache
def get_settings() -> WorkerSettings:
    return WorkerSettings()
