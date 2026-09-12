# Code structure

```
ml-cv-learning/
  worker/
    app/
      main.py              CLI: --source + --zones
      camera_loop.py       one camera: pull → detect → features
      config.py            env flags (INFERENCE_FPS, FEATURE_*)
      models.py            ZoneConfig
      zones.py             point-in-polygon, bbox vs zone
      zone_loader.py       YAML → ZoneConfig
      footfall.py          entrance crossing + dwell
      inference/
        backend.py         Detection dataclass
        yolo_backend.py    YOLOv8n + ByteTrack
        pose_backend.py    MediaPipe BlazePose + helpers
      rtsp/
        puller.py          file / webcam / RTSP frames
      features/
        demographics.py    gender / age on an enter crop (example)
        queue_length.py    people inside a queue zone
        kit_detection.py   object-in-zone stub (extend this)
    requirements.txt
  testdata/
    zones.yaml             example polygons (0–1 coords)
  FEATURE_CONTRACT.md
  PROMPT.md
```

There is no API, database, or web UI. Features log results. That is enough to iterate on models.

## Frame loop

`CameraLoop._process_frame` is the only place features are called.

```
frame
  → YoloBackend.detect() or detect_tracked()
  → optional PoseBackend.estimate() per person crop
  → EntranceCounter.update()          if FEATURE_FOOTFALL
      → demographics.estimate()       if FEATURE_DEMOGRAPHICS (per enter only)
  → QueueMonitor.estimate()           if FEATURE_QUEUE
  → kit_detection.estimate()          if FEATURE_KIT
```

Rules for this file:

- Do not put model code here.
- Each feature is a lazy import + flag.
- A feature must not crash the loop. Catch inside the module.

## Detection (`YoloBackend`)

- Model: `yolov8n.pt` (nano, CPU).
- `detect(frame)` — boxes for the current frame. No IDs.
- `detect_tracked(frame)` — same plus `track_id` via ByteTrack (`persist=True`).
- Output: `Detection(class_name, confidence, x1, y1, x2, y2, track_id)`.

Use tracked mode when you need identity across time (entrance count, ReID experiments). Use plain detect for occupancy / object presence.

COCO already includes `person`, bags, bottles, laptops, and similar. Custom classes need a different `.pt` — keep that swap inside `yolo_backend.py` or a new backend.

## Tracking (ByteTrack)

ByteTrack associates boxes across frames. IDs are **not** people forever:

- A person who leaves and comes back often gets a new ID.
- Occlusion can split one person into two IDs.

`footfall.py` already has:

- same-frame NMS (IoU ≥ 0.40)
- ID-split guard (do not count a new ID that overlaps an already-inside track)

If you add ReID later, do it as `features/reid.py` and call it on enter / exit, not on every frame.

## Pose (MediaPipe BlazePose)

File: `worker/app/inference/pose_backend.py`.

- Input: BGR crop of one person (not the full frame).
- Output: 33 `Keypoint(x, y, z, visibility)` in **crop-normalised 0–1**.
- `model_complexity=0` — fastest CPU setting.
- Lazy: constructed only when `FEATURE_POSE=true`.

Landmark indices (subset):

| Index | Joint |
|---|---|
| 0 | nose |
| 11 / 12 | shoulders |
| 13 / 14 | elbows |
| 15 / 16 | wrists |
| 23 / 24 | hips |
| 25 / 26 | knees |
| 27 / 28 | ankles |

Helpers already on `PoseBackend`:

- `is_hand_near_head` — wrist close to nose
- `is_horizontal` — lying-down heuristic
- `is_seated` — hip and knee at similar height

Pose is expensive relative to YOLO. Turn it on only for features that need skeleton (fall, seated, phone-to-ear). Queue length and kit detection should **not** require pose.

## Zones

`ZoneConfig`: `zone_id`, `zone_type`, `polygon`, `name`.

Preferred polygon format: **normalised 0–1**. `polygon_to_frame_space` scales them to the current frame.

Built-in types the loop already understands:

| type | Used by |
|---|---|
| `entrance` | `footfall.EntranceCounter` |
| `queue` | `features.queue_length` |
| `kit` | `features.kit_detection` |

Add new types freely (`counter`, `restricted`, `aisle`). A feature should filter `zones` by `zone_type` itself.

## Existing features

**Footfall** — unique enters into an `entrance` zone using the feet point of a tracked person. Emits `CrossingEvent` (bbox + track_id). Also emits dwell when the track disappears.

**Queue** — count of person boxes whose centre is inside a `queue` zone this frame. Peak is tracked for the run.

**Demographics** — InsightFace on a padded person crop, once per enter. Returns gender / age / kid-vs-adult or unknowns. Never changes the count.

**Kit** — stub: any COCO class in `DEFAULT_KIT_CLASSES` whose box overlaps a `kit` zone. Replace the class list or the model when you do real kit / PPE work.

## What this lab is not

- No persistence (no Postgres, no object storage).
- No clips, push notifications, or multi-tenant loader.
- No product UI.

Keep it that way so this repo stays a gym for algorithms.
