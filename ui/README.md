# CV Pipeline Lab — local test UI

A local Streamlit tool to build zones, run the pipeline on an MP4, and view
per-signal screenshots (people entering, groups, kids, exits).
**Local only — this folder is gitignored and not pushed.**

## Install

Uses the same virtualenv as the worker (so it can import `app`).

```bash
cd ..           # repo root
source worker/.venv/bin/activate
pip install -r ui/requirements.txt
```

## Run

```bash
cd ui
streamlit run app_ui.py
```

Then in the browser:

1. **Zone Builder** — paste the MP4 path, draw rectangle zones (staff / queue /
   kit / exclude), set the entrance tripwire line + inside direction, and
   **Save zones.yaml**.
2. **Run** — pick flags (footfall / groups / age-gender / staff filter) and hit
   **Run**. The first run downloads YOLO11n and InsightFace `buffalo_l`.
3. **Results** — counts, age/gender charts, and screenshot galleries captured
   at the moment each signal fired.

## Notes

- The guard/staff exclusion needs a `staff` (or `exclude`) zone drawn over the
  guard's spot; anyone passing through it is not counted.
- The entrance is a **directional line**: someone stepping in and back out is
  not double-counted.
- Outputs (screenshots, zones.yaml, summary.json) go under `ui/outputs/` and
  are gitignored.
