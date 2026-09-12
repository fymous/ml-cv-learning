"""MediaPipe BlazePose — runs on a cropped person bbox, returns 33 keypoints.

Used only when FEATURE_POSE=true (or a feature that needs pose is on).
Complexity 0 = fastest CPU path. Coordinates are normalised 0–1 inside the crop.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Keypoint:
    x: float  # normalised 0-1 within the crop
    y: float
    z: float
    visibility: float


# MediaPipe landmark indices
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
MP_NOSE = 0
MP_LEFT_EYE_INNER = 1
MP_LEFT_EYE = 2
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_INNER = 4
MP_RIGHT_EYE = 5
MP_RIGHT_EYE_OUTER = 6
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_THUMB = 21
MP_RIGHT_THUMB = 22
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32


class PoseBackend:
    def __init__(self) -> None:
        import mediapipe as mp  # lazy import — heavy

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,  # fastest
            min_detection_confidence=0.5,
        )

    def estimate(self, frame_crop: np.ndarray) -> list[Keypoint] | None:
        """Run pose on a BGR crop. Returns None if no person detected."""
        import cv2

        rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        return [
            Keypoint(lm.x, lm.y, lm.z, lm.visibility)
            for lm in result.pose_landmarks.landmark
        ]

    def close(self) -> None:
        self._pose.close()

    @staticmethod
    def is_hand_near_head(kps: list[Keypoint]) -> bool:
        """Wrist close to nose — phone / face-touch heuristic."""
        nose = kps[MP_NOSE]
        for wrist_idx in (MP_LEFT_WRIST, MP_RIGHT_WRIST):
            w = kps[wrist_idx]
            if w.visibility < 0.5:
                continue
            dist = ((w.x - nose.x) ** 2 + (w.y - nose.y) ** 2) ** 0.5
            if dist < 0.15:
                return True
        return False

    @staticmethod
    def is_horizontal(kps: list[Keypoint]) -> bool:
        """Person lying down (shoulder–hip vector mostly horizontal)."""
        ls = kps[MP_LEFT_SHOULDER]
        rs = kps[MP_RIGHT_SHOULDER]
        lh = kps[MP_LEFT_HIP]
        rh = kps[MP_RIGHT_HIP]
        mid_shoulder_y = (ls.y + rs.y) / 2
        mid_hip_y = (lh.y + rh.y) / 2
        vertical_span = abs(mid_hip_y - mid_shoulder_y)
        mid_shoulder_x = (ls.x + rs.x) / 2
        mid_hip_x = (lh.x + rh.x) / 2
        horizontal_span = abs(mid_hip_x - mid_shoulder_x)
        return horizontal_span > vertical_span

    @staticmethod
    def is_seated(kps: list[Keypoint]) -> bool:
        """Hip–knee height similar in the crop → sitting heuristic."""
        for hip_idx, knee_idx in ((MP_LEFT_HIP, MP_LEFT_KNEE), (MP_RIGHT_HIP, MP_RIGHT_KNEE)):
            hip = kps[hip_idx]
            knee = kps[knee_idx]
            if hip.visibility < 0.5 or knee.visibility < 0.5:
                continue
            if abs(knee.y - hip.y) < 0.1:
                return True
        return False
