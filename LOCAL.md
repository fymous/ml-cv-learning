# Run this repo locally

No Docker. No Podman. No database. No cloud.

This lab is **Python + a video file** (or a webcam). That is enough to develop features on another laptop.

---

## What you need

| Requirement | Notes |
|---|---|
| Python 3.11 or 3.12 | 3.13 can break MediaPipe / InsightFace — avoid it |
| ~4 GB free disk | first run downloads `yolov8n.pt` (~6 MB) and pip wheels |
| 8 GB RAM | comfortable; 16 GB better if you turn pose + demographics on |
| A short `.mp4` | your own or a public clip — videos are gitignored |

You do **not** need: Podman, Docker Desktop, Postgres, MinIO, Redis, or an account for any cloud.

---

## 1. Clone and venv (Windows)

PowerShell:

```powershell
git clone https://github.com/fymous/ml-cv-learning.git
cd ml-cv-learning\worker
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy ..\.env.example ..\.env
```

If `Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

macOS / Linux:

```bash
cd ml-cv-learning/worker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env
```

---

## 2. Add a sample video

Put a clip at `testdata/sample.mp4` (any name is fine). Indoor walking / a doorway works best.

Edit `testdata/zones.yaml` so the polygons roughly match that clip. Coordinates are **0–1** (`[x, y]`).

---

## 3. Run one pass

From the `worker/` folder, with the venv active:

```powershell
python -m app.main --source ..\testdata\sample.mp4 --zones ..\testdata\zones.yaml
```

Webcam:

```powershell
python -m app.main --source 0
```

RTSP (if you have a camera on the LAN):

```powershell
python -m app.main --source "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101" --zones ..\testdata\zones.yaml
```

`WORKER_MODE=once` (default) reads a file to the end and exits. Logs print enters, queue counts, and any kit hits.

First run downloads **YOLOv8n**. Later runs reuse the cached `.pt` in the working directory.

---

## 4. Feature flags

Edit `.env` in the **repo root** (`ml-cv-learning/.env`):

```
INFERENCE_FPS=1.0
FEATURE_FOOTFALL=true
FEATURE_QUEUE=true
FEATURE_DEMOGRAPHICS=false
FEATURE_KIT=false
FEATURE_POSE=false
```

| Flag | What it does | Cost |
|---|---|---|
| `FEATURE_FOOTFALL` | count enters on `entrance` zones | cheap |
| `FEATURE_QUEUE` | people inside `queue` zones | cheap |
| `FEATURE_KIT` | COCO objects in a `kit` zone | cheap |
| `FEATURE_POSE` | BlazePose on each person crop | heavier CPU |
| `FEATURE_DEMOGRAPHICS` | gender/age on each enter | downloads ~160 MB once |

Turn on only what you are testing.

---

## 5. Cursor on this laptop

1. Open the `ml-cv-learning` folder in Cursor (not any other repo).
2. New chat → paste all of `PROMPT.md`.
3. Last line: the feature from `FEATURES.md`, e.g. `The feature I want is: queue wait estimate.`

---

## Common problems

| Symptom | Fix |
|---|---|
| `No module named app` | Run from `worker/`: `python -m app.main ...` |
| `could not open source` | Path to the mp4 is wrong; use `..\testdata\...` from `worker/` |
| MediaPipe / InsightFace install fails | Use Python 3.11 or 3.12, not 3.13 |
| Very slow | Keep `INFERENCE_FPS=1.0`; leave pose and demographics off |
| Zones never fire | Polygons don’t match the clip — redraw `zones.yaml` |
| Execution policy error | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## Why there are no containers

A product stack needs Postgres, object storage, and an API. This lab does not. Adding Podman here would only install unused services and hide import errors.

If you later need a repeatable environment, a venv + `requirements.txt` is the environment. Do not add Compose unless a feature truly needs a service (it shouldn’t).
