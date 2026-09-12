"""Frame puller for local files, webcam, RTSP, or HTTP video.

Supports:
  rtsp://   — IP camera / NVR
  file://   — local video file (or a bare path that exists)
  http(s):// — remote video file
  0, 1 …   — webcam index
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    RTSP = "rtsp"
    FILE = "file"
    HLS = "hls"
    WEBCAM = "webcam"


_VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")


def _detect_source_type(url: str) -> SourceType:
    if url.startswith("rtsp://") or url.startswith("rtsps://"):
        return SourceType.RTSP
    if url.startswith("file://") or Path(url).exists():
        return SourceType.FILE
    if url.startswith("http://") or url.startswith("https://"):
        path = url.split("?", 1)[0].lower()
        if any(path.endswith(s) for s in _VIDEO_SUFFIXES):
            return SourceType.FILE
        return SourceType.HLS
    if url.isdigit():
        return SourceType.WEBCAM
    return SourceType.RTSP


class FramePuller:
    """Pull frames from any video source at a configurable FPS."""

    def __init__(
        self,
        url: str,
        fps: float | None = None,
        camera_id: str = "lab",
        *,
        once: bool | None = None,
    ) -> None:
        self.url = url
        self.camera_id = camera_id
        settings = get_settings()
        self._fps = fps or settings.inference_fps
        self._interval = 1.0 / self._fps
        self._reconnect_delay = settings.rtsp_reconnect_delay
        self._max_retries = settings.rtsp_max_retries
        self._source_type = _detect_source_type(url)
        self._cap: cv2.VideoCapture | None = None
        if once is None:
            once = settings.worker_mode.lower() == "once"
        self._once = once
        self._once_max_sec = settings.worker_once_max_sec if once else 0.0

    def _open(self) -> cv2.VideoCapture:
        actual_url: str | int = self.url
        if self._source_type == SourceType.FILE:
            actual_url = self.url.replace("file://", "")
        elif self._source_type == SourceType.WEBCAM:
            actual_url = int(self.url)

        cap = cv2.VideoCapture(actual_url)

        if self._source_type == SourceType.RTSP:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000)

        return cap

    def frames(self):
        """Yields (frame, wall_ts, media_ts).

        live mode: reconnects on failure; file sources loop.
        once mode: file EOF ends; live disconnect / max_sec ends.
        File+once samples along the clip (~inference_fps of media time).
        """
        retries = 0
        started = time.monotonic()

        while True:
            if self._once and self._once_max_sec > 0:
                if time.monotonic() - started >= self._once_max_sec:
                    logger.info(
                        "camera=%s once max_sec=%.1f reached — stopping",
                        self.camera_id,
                        self._once_max_sec,
                    )
                    return

            logger.info(
                "camera=%s source=%s connecting once=%s",
                self.camera_id,
                self._source_type.value,
                self._once,
            )
            cap = self._open()

            if not cap.isOpened():
                retries += 1
                logger.warning("camera=%s failed to open (attempt %d)", self.camera_id, retries)
                if self._once or (self._max_retries and retries > self._max_retries):
                    raise RuntimeError(f"Camera {self.camera_id}: could not open source")
                time.sleep(self._reconnect_delay)
                continue

            logger.info("camera=%s connected", self.camera_id)
            retries = 0
            last_yield = 0.0
            file_index = 0
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            file_stride = 1
            if self._source_type == SourceType.FILE and self._once and src_fps > 0:
                file_stride = max(1, int(round(src_fps / self._fps)))
                logger.info(
                    "camera=%s file once stride=%d src_fps=%.1f inference_fps=%.1f",
                    self.camera_id,
                    file_stride,
                    src_fps,
                    self._fps,
                )

            try:
                while True:
                    if self._once and self._once_max_sec > 0:
                        if time.monotonic() - started >= self._once_max_sec:
                            return

                    mono = time.monotonic()
                    ok, frame = cap.read()

                    if not ok:
                        if self._source_type == SourceType.FILE:
                            if self._once:
                                logger.info("camera=%s file EOF — once pass complete", self.camera_id)
                                return
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        logger.warning("camera=%s read failed", self.camera_id)
                        if self._once:
                            return
                        break

                    if self._source_type == SourceType.FILE and self._once:
                        take = (file_index % file_stride) == 0
                        file_index += 1
                        if not take:
                            continue
                    elif mono - last_yield < self._interval:
                        if self._source_type == SourceType.FILE:
                            time.sleep(self._interval - (mono - last_yield))
                            mono = time.monotonic()
                        else:
                            continue

                    last_yield = mono
                    wall_ts = time.time()
                    if self._source_type == SourceType.FILE:
                        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                        media_ts = max(0.0, pos_msec / 1000.0)
                    else:
                        media_ts = wall_ts
                    yield frame, wall_ts, media_ts

            finally:
                cap.release()

            if self._once:
                return
            time.sleep(self._reconnect_delay)

    def snapshot(self) -> np.ndarray | None:
        try:
            cap = self._open()
            if not cap.isOpened():
                return None
            ok, frame = cap.read()
            cap.release()
            return frame if ok else None
        except Exception:
            return None
