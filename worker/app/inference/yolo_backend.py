"""YOLOv8n CPU backend using Ultralytics (auto-downloads model on first run)."""

import numpy as np

from app.config import get_settings
from app.inference.backend import Detection, InferenceBackend


class YoloBackend(InferenceBackend):
    def __init__(self, model_path: str | None = None, conf: float | None = None) -> None:
        from ultralytics import YOLO

        settings = get_settings()
        self._conf = conf if conf is not None else settings.yolo_conf_threshold
        self._tracker = settings.tracker
        self._model = YOLO(model_path or settings.yolo_model)
        self._names: dict[int, str] = self._model.names  # type: ignore[assignment]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, conf=self._conf, verbose=False)
        return self._parse_results(results, tracked=False)

    def detect_tracked(self, frame: np.ndarray) -> list[Detection]:
        """Tracked detection — stable IDs across frames for crossing counts.

        Tracker is configurable (`TRACKER`, default botsort.yaml). BoT-SORT uses
        appearance/ReID cues, so it keeps IDs stable across brief occlusion much
        better than motion-only ByteTrack — which directly reduces the
        same-person-counted-twice error at entrances.
        """
        results = self._model.track(
            frame,
            conf=self._conf,
            persist=True,
            verbose=False,
            tracker=self._tracker,
        )
        return self._parse_results(results, tracked=True)

    def _parse_results(self, results, *, tracked: bool) -> list[Detection]:
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = None
                if tracked and box.id is not None:
                    track_id = int(box.id[0])
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self._names.get(cls_id, str(cls_id)),
                        confidence=float(box.conf[0]),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        track_id=track_id,
                    )
                )
        return detections
