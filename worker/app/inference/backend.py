"""Abstract inference backend — swappable between CPU ONNX, OpenVINO, TensorRT."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def iou(self, other: "Detection") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (
            (self.x2 - self.x1) * (self.y2 - self.y1)
            + (other.x2 - other.x1) * (other.y2 - other.y1)
            - inter
        )
        return inter / union if union > 0 else 0.0


class InferenceBackend(ABC):
    """Stateless inference backend. Implement detect() for each hardware target."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a BGR frame. Return list of detections."""
        ...

    def detect_tracked(self, frame: np.ndarray) -> list[Detection]:
        """Optional: detection with stable track IDs (defaults to detect())."""
        return self.detect(frame)

    def warmup(self, frame: np.ndarray) -> None:
        """Optional: run one inference to warm up the model."""
        self.detect(frame)
