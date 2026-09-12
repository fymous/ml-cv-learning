# ml-cv-learning

Local computer-vision lab. Frame in → detections / tracks / optional pose → optional feature modules out.

This repo is for building new capabilities on a laptop (queue length, kit / object presence, pose rules, ReID, and so on). There is no cloud, auth, or dashboard here.

## Tech stack

| Layer | Library | Role |
|---|---|---|
| Detect | Ultralytics **YOLOv8n** | People + COCO objects, CPU |
| Track | **ByteTrack** (via Ultralytics `.track`) | Stable `track_id` across frames |
| Pose | **MediaPipe BlazePose** | 33 keypoints on a person crop |
| Face / age | **InsightFace buffalo_s** | Optional gender + age on an enter crop |
| Runtime | OpenCV + ONNX Runtime | Frames + CPU inference |
| Config | pydantic-settings + YAML | Env flags + zone polygons |

Default inference rate is **1 FPS**. That is enough for crossing counts and zone occupancy. Pose and face models run only when their flags are on.

## Quick start

```bash
git clone https://github.com/fymous/ml-cv-learning.git
cd ml-cv-learning/worker
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env
```

Drop a sample `.mp4` into `testdata/`, then:

```bash
python -m app.main --source ../testdata/your-clip.mp4 --zones ../testdata/zones.yaml
```

Webcam:

```bash
python -m app.main --source 0
```

First run downloads `yolov8n.pt`. InsightFace downloads `buffalo_s` only if `FEATURE_DEMOGRAPHICS=true`.

## Feature flags

Each capability is independent. Flip them in `.env`:

```
FEATURE_FOOTFALL=true
FEATURE_QUEUE=true
FEATURE_DEMOGRAPHICS=false
FEATURE_KIT=false
FEATURE_POSE=false
```

## Docs

- [STRUCTURE.md](STRUCTURE.md) — folders, loop, YOLO, ByteTrack, pose, zones
- [FEATURE_CONTRACT.md](FEATURE_CONTRACT.md) — how to add a capability so it can be copied elsewhere later
- [PROMPT.md](PROMPT.md) — paste this into Cursor on another machine when you start a new feature
