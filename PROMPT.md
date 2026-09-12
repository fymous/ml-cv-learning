# Cursor prompt — new CV capability

Paste this as the first message in a new chat on the laptop that has this repo. Then add one sentence naming the feature you want.

---

You are working in the **ml-cv-learning** repo. It is a local computer-vision lab only.

Read these files before writing code:

- `STRUCTURE.md` — architecture, YOLO, ByteTrack, pose, zones
- `FEATURE_CONTRACT.md` — how every capability must be shaped
- `worker/app/camera_loop.py` — the only place features are called
- `worker/app/config.py` — feature flags
- `worker/app/features/` — existing modules (copy this style)
- `worker/app/inference/pose_backend.py` — 33 BlazePose landmarks + helpers
- `worker/app/inference/yolo_backend.py` — detect / detect_tracked
- `worker/app/footfall.py` — `CrossingEvent` if you need an enter bbox

## Hard rules

1. Do **not** add cloud, auth, databases, dashboards, or deploy scripts.
2. Do **not** mention any company, product, or customer name in code or docs.
3. One new feature = one file in `worker/app/features/<name>.py` + one boolean flag + a short hook in `camera_loop.py`.
4. `estimate(...)` must fail soft (never crash the loop). Lazy-load models.
5. Prefer CPU. Default 1 FPS. Do not raise FPS unless the feature truly needs it.
6. Use existing detections / tracks / zones first. Add a new model only if YOLO + pose cannot do the job.
7. Pose is expensive. Turn `FEATURE_POSE` on only if the feature needs a skeleton.
8. Keep comments factual. No marketing language.
9. After the feature works, tell me exactly which files to copy into another codebase (the backport list).

## How to implement a feature

1. Name it (`queue_wait`, `ppe`, `reid`, `loitering`, …).
2. Decide the trigger:
   - every frame (occupancy, objects in zone)
   - per enter (`CrossingEvent` from footfall)
   - pose-based (fall, seated, hand-to-head)
3. Add `feature_<name>: bool = False` in `config.py` and `.env.example`.
4. Write `features/<name>.py` with a dataclass result + `estimate(...)`.
5. Hook it in `camera_loop.py` behind the flag. Log the result.
6. If you need a new zone type, document it in `testdata/zones.yaml` and `STRUCTURE.md`.
7. If you need a sample video, put it in `testdata/` (videos are gitignored).

## Feature ideas (pick one per chat)

- **Queue length / wait** — improve `queue_length.py`: dwell in zone, order of people, wait estimate.
- **Kit / object detection** — replace the COCO stub in `kit_detection.py` (custom classes, PPE helmet/vest, tools).
- **Restricted-area** — person centre inside a `restricted` zone for N seconds.
- **Staffed counter** — two zones (`customer_area`, `staff_area`); customer present + staff absent.
- **Loitering** — same track inside a zone longer than T seconds.
- **ReID / re-entry** — embedding on exit, match on next enter (do not increment count if same person within a window).
- **Fall / lying down** — `FEATURE_POSE` + `PoseBackend.is_horizontal`, sustain for N frames.
- **Seated vs standing** — `PoseBackend.is_seated`.
- **Phone-to-ear** — `PoseBackend.is_hand_near_head`.
- **Group / together** — tracks whose boxes are close when they enter.
- **After-hours motion** — person detected outside a time window (use wall clock, no cloud).
- **Line crossing (in + out)** — extend footfall so `direction` can be `exit` and net occupancy = enters − exits.
- **Headcount heatmap** — accumulate feet-points, write a local numpy / PNG at end of run.

## When I name a feature

Implement it end to end in this repo. Run mentally through empty zones, no detections, model-missing, and overlapping boxes. Then summarise:

- what you added
- how to turn it on
- accuracy / FPS caveats
- files to copy later

Start now. The feature I want is:

<!-- write the feature name on the next line -->

---

## Short follow-up prompts (optional)

After the first feature works, you can send just:

```
Add FEATURE_LOITERING: same track_id inside a zone for more than 20 seconds. Log when it trips. Fail soft. No new dependencies.
```

```
Improve kit_detection to use a custom YOLO .pt if YOLO_KIT_MODEL is set, else keep COCO. Document the env var.
```

```
Add exit direction to footfall and a net occupancy per entrance zone. Do not break CrossingEvent fields already used by demographics.
```
