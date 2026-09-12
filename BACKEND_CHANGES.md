# Backend Changes — CV Analytics Worker

This documents everything changed in `worker/` (plus root config files it
depends on: `.env.example`, `.gitignore`, `testdata/zones.yaml`) during this
round of work. **UI (`ui/`) is intentionally excluded** — it's a local,
gitignored test harness, not part of the shipped backend.

## Why

The existing worker had three concrete accuracy bugs, plus two missing
capabilities:

1. **Double counting** — the entrance was a polygon; anyone who stepped in and
   immediately back out was counted as 2 (enter + "enter again" on re-sight).
2. **No exit / occupancy** — footfall was enter-only, no net occupancy.
3. **No demographics granularity** — "kid vs adult" only, no gender/age
   breakdown, and a single-shot face read at the doorway rarely found a usable
   face (people are often not facing the camera at the exact enter moment).
4. **No group-vs-individual distinction** — a family of 3 counted as 3 separate
   visits.
5. **No way to exclude staff/guards/passers-by** from the visitor count.

Everything below addresses one of these five.

---

## 1. Detection & tracking model upgrades

| Setting | Before | After | File |
|---|---|---|---|
| Detector | `yolov8n.pt` | `yolo11n.pt` | `worker/app/config.py` |
| Tracker | `bytetrack.yaml` (hardcoded) | `botsort.yaml` (configurable via `TRACKER` env) | `worker/app/config.py`, `worker/app/inference/yolo_backend.py` |
| Inference rate | `1.0` fps | `2.0` fps (default; UI test runs use 5 fps) | `worker/app/config.py` |

**Why BoT-SORT over ByteTrack:** ByteTrack is motion-only — under brief
occlusion in a crowd (people crossing paths at a doorway) it frequently drops
and re-mints a new `track_id` for the same person, which both breaks the
directional line-crossing logic (a "crossing" needs a continuous trajectory
for one ID) and pollutes demographics voting. BoT-SORT adds appearance/ReID
cues on top of motion, so it re-attaches the same ID through short occlusions
much more reliably.

```12:29:worker/app/inference/yolo_backend.py
        self._tracker = settings.tracker
        self._model = YOLO(model_path or settings.yolo_model)
        ...
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
```

A second, more aggressive tracker profile was added for busy/static CCTV
doorways: `worker/app/trackers/botsort_persist.yaml`. It raises
`new_track_thresh` (fewer spurious short-lived tracks), raises `track_buffer`
to 60 frames (keeps a lost track "alive" longer so occlusion re-attaches the
same ID instead of minting a new one), disables `gmc_method` (global motion
compensation assumes a moving camera — CCTV is static), and enables
`with_reid`. This is opt-in via `TRACKER=app/trackers/botsort_persist.yaml`;
the stock `botsort.yaml` remains the default.

`lap>=0.5.12` was added to `worker/requirements.txt` — BoT-SORT's assignment
step needs it and Ultralytics doesn't pull it in automatically.

---

## 2. Entrance counting rewrite — directional line-crossing (`footfall.py`)

This is the core fix for double-counting. `worker/app/footfall.py` was
rewritten around a **directed tripwire line** instead of "is the point inside
this polygon":

- A zone can now optionally carry `line: [[ax,ay],[bx,by]]` (two points the
  visitor walks across) and `in_direction: [dx,dy]` (a vector pointing into
  the venue).
- Each tracked person's **feet-point** trajectory (`(prev, current)` per
  frame) is tested for **segment intersection** against that line
  (`app/zones.py::segments_intersect`).
- If it crosses, the direction (enter vs exit) is derived from whether the
  movement vector agrees with `in_direction`
  (`app/zones.py::crossing_is_inward`) — not from "am I now inside a shape."

Why this kills the bug: a person who steps in and immediately steps back out
crosses the same line **twice**, once inward (enter) and once outward (exit) —
net occupancy correctly nets to 0, instead of the old polygon logic counting
"newly inside" twice because of how re-sight/ID-churn worked.

The **polygon mode still exists** as a fallback for zones with no `line`
defined (backward compatible with old `zones.yaml` files), using
first-sight-inside + outside→inside transition logic.

Both modes share the same guardrails, tightened compared to before:

| Guardrail | Purpose |
|---|---|
| `_NMS_IOU = 0.40` — NMS across overlapping person boxes pre-count | Stops the same physical person from being double-detected and double-counted in one frame |
| `_MIN_TRACK_FRAMES = 2` | A brand-new track ID cannot count on its very first sighting (kills spurious "already across the line" false enters from a track that just appeared) |
| `_COUNT_COOLDOWN_SEC = 1.0` | Same track cannot register a second crossing within 1s (debounces jitter at the line) |
| `_SPLIT_IOU = 0.50` (polygon mode) | Detects when the tracker split one physical person into two IDs, so the split ID doesn't double-count |
| `_MISS_GRACE_FRAMES = 12` | A track can go briefly undetected without losing its state (handles a frame or two of missed detection without falsely triggering exit/dwell logic) |

`EntranceCounter.update(...)` now also accepts `exclude_ids` — track IDs
(staff/guards, see §4) are still tracked for state continuity but **never
counted or emitted as a `CrossingEvent`**.

`CrossingEvent.direction` changed from an implicit "always an entry" concept
to an explicit `"enter" | "exit"` string, and `ZoneFootfall` now tracks
`entered`, `exited`, and derives `net = entered - exited` — giving you actual
occupancy, not just cumulative arrivals.

New geometry primitives backing this live in `worker/app/zones.py`:
`line_to_frame_space`, `_orient` (signed triangle area, for orientation
tests), `segments_intersect` (proper segment-intersection test),
`crossing_is_inward` (direction resolution).

---

## 3. Group counting — new feature (`features/group_count.py`)

New module, `GroupCounter`, wired in behind `FEATURE_GROUP` (default **on**).

It consumes the `enter` `CrossingEvent`s that `EntranceCounter` already
produces (so it automatically inherits the fixed, de-duplicated, directional
counts above — no separate detection pass) and clusters them purely
geometrically:

> A new enter joins an existing **open** group if it happened within
> `window_sec` (4.0s) of that group's last-joined member **and** its
> feet-point is within `radius_frac × frame_width` (0.18) of the group's
> running centroid. Otherwise it opens a new group of size 1.

```60:122:worker/app/features/group_count.py
class GroupCounter:
    def __init__(
        self, window_sec: float = _WINDOW_SEC, radius_frac: float = _RADIUS_FRAC
    ) -> None:
        ...
    def add(
        self, event: CrossingEvent, media_ts: float, frame_w: int, frame_h: int
    ) -> GroupResult | None:
        """Assign one enter event to a group. Returns the group it landed in."""
```

No extra model and no per-frame cost — it's stateful bookkeeping on top of
crossing events, and it directly benefits from the tracker upgrade (§1) since
it depends on stable `track_id`s to avoid double-joining the same physical
person to a group twice.

`summary()` reports both the raw cluster count and the **honest "real
group" metric**, split out on purpose:

```123:137:worker/app/features/group_count.py
        multi = [s for s in sizes if s >= 2]
        return {
            # every distinct arrival cluster, including solo arrivals
            "total_clusters": self.total_groups,
            # genuine groups = 2+ people arriving together
            "groups_2plus": len(multi),
            "people_in_groups": sum(multi),
            "total_people": sum(sizes),
            "avg_group_size": round(sum(multi) / len(multi), 2) if multi else 0.0,
            "size_histogram": dict(sorted(histogram.items())),
        }
```

**Known limitation (flagged, not hidden):** co-arrival-in-time-and-space is a
heuristic, not a true "are these people socially together" signal — two
strangers who happen to walk in at the same moment from the same side will be
merged into one group, and members of a real group who arrive staggered by
more than 4s / spread wider than the radius will be split into separate
groups. There is no cheaper way to do this without adding a re-ID/appearance
model; this was a conscious trade-off given the "keep pricing balanced"
constraint. Only entries are grouped — a family leaving together is not
currently tracked as a group on exit.

---

## 4. Staff / passer-by exclusion — new feature (`features/staff_filter.py`)

New module, `StaffFilter`, wired in behind `FEATURE_STAFF_FILTER` (default
**on**). Two independent, cheap mechanisms — no extra heavy model:

1. **Zone-based (primary, deterministic).** Any `ZoneConfig` with
   `zone_type` `"staff"` or `"exclude"` defines a polygon. The first time a
   track's feet-point enters it, that `track_id` is tagged staff **sticky
   forever** (`self._staff_ids`, a monotonically growing set) — even if it
   later walks through the entrance normally, it stays excluded.
2. **Uniform colour (optional supplement, off by default).** If
   `STAFF_UNIFORM_HSV` is set (JSON list of `[h_lo,s_lo,v_lo,h_hi,s_hi,v_hi]`
   OpenCV-HSV ranges), a fresh track's torso ROI (15–55% down the box,
   middle 40% across) median HSV is checked against each range. This is a
   heuristic — a visitor wearing a similar colour can false-match — so it's
   meant as a supplement to a zone, not a replacement.

```59:88:worker/app/features/staff_filter.py
    def mark(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        frame=None,
    ) -> set[int]:
        """Update and return the cumulative set of staff/excluded track ids."""
```

`mark()` is called once per frame in `camera_loop.py` and its returned set is
threaded through to both `EntranceCounter.update(exclude_ids=...)` (so staff
never count toward footfall) and the demographics observer (no point running
a face read on a guard, §5).

**Known limitation:** this only excludes people while they're inside the
drawn zone; it cannot re-identify "the same guard" once they leave that zone
and later cross the entrance from elsewhere without appearance embeddings.
That's called out in the module docstring as a deliberate scope cut
(`features/reid.py`, not built).

---

## 5. Demographics overhaul — multi-frame voting + age buckets (`features/demographics.py`)

The whole approach changed from **single-shot** to **accumulate-then-vote**:

- **Before:** one InsightFace read on the crop at the exact enter frame. If
  the person's face wasn't visible/frontal at that instant (very common at a
  doorway — people often look down or sideways while walking through), the
  read silently failed.
- **After:** `DemographicsAggregator.observe(frame, det)` is called **every
  frame** a track is visible (for up to `_MAX_VOTES = 7` good reads per
  track), and `.result(track_id)` — called once, on the `enter` event — takes
  the **majority-vote gender** and **median age** across all accumulated
  votes for that track.

A quality gate discards low-value reads before they ever become a vote:
`_MIN_DET_SCORE = 0.60` (face detector confidence) and `_MIN_FACE_PX = 24`
(minimum face box dimension) — back-of-head, tiny, or low-confidence face
detections are ignored rather than polluting the vote.

```81:100:worker/app/features/demographics.py
def _get_analyzer():
    ...
            from insightface.app import FaceAnalysis
            pack = get_settings().demographics_model
            app = FaceAnalysis(
                name=pack,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "genderage"],
            )
```

Model pack changed from hardcoded `buffalo_s` to configurable
`DEMOGRAPHICS_MODEL` (default now `buffalo_l` — noticeably more accurate at a
modest extra CPU cost; still CPU-only via `onnxruntime`, `providers=
["CPUExecutionProvider"]`). Still fails soft — no model, no face, or an
inference exception all resolve to `None`/unknown fields rather than raising.

**Age buckets replace the old binary kid/adult split:**

```37:51:worker/app/features/demographics.py
AGE_BUCKETS: list[tuple[int, int, str]] = [
    (0, 12, "0-12"),
    (13, 19, "13-19"),
    (20, 34, "20-34"),
    (35, 54, "35-54"),
    (55, 200, "55+"),
]
```

New `DemographicsTally` class accumulates a **run-level distribution**
(`by_age_group`, `by_gender`, `by_age_gender`, `counted`, `with_reading`) over
every counted visitor — this is what feeds the age/gender breakdown, not just
a per-person label.

`prune(active_ids)` bounds memory: a track's accumulated votes are dropped
after `_PRUNE_MISS = 30` frames of not being seen, so a long-running (live)
process doesn't leak memory holding vote lists for people who left the scene.

The old single-shot `estimate(frame, event)` function is kept for backward
compatibility but is no longer the code path `camera_loop.py` uses.

---

## 6. Zone model & config plumbing

`worker/app/models.py` — `ZoneConfig` gained two optional fields for the
line-crossing entrance mode, and `polygon` became optional (a `line`-only
entrance zone no longer needs a redundant polygon):

```1:20:worker/app/models.py
@dataclass
class ZoneConfig:
    zone_id: str
    zone_type: str
    polygon: list[list[float]] = field(default_factory=list)
    name: str = ""
    line: list[list[float]] | None = None
    in_direction: list[float] | None = None
```

`worker/app/zone_loader.py` parses `line`/`in_direction` out of the YAML if
present, defaulting to `None` (falls back to polygon mode).

`worker/app/config.py` — new settings: `tracker` (`TRACKER` env, default
`botsort.yaml`), `demographics_model` (`DEMOGRAPHICS_MODEL` env, default
`buffalo_l`), `feature_group` (default `True`), `feature_staff_filter`
(default `True`), `staff_uniform_hsv` (`STAFF_UNIFORM_HSV` env, JSON list,
default `[]`).

`testdata/zones.yaml` updated to demonstrate the new line-based entrance +
a `staff` exclusion zone, with inline comments explaining the format.

---

## 7. `camera_loop.py` — orchestration changes

Per-frame order of operations changed to support the new features, all still
individually flag-gated:

1. Detect (tracked, if footfall/group/demographics needs stable IDs — this
   condition was widened from "just footfall" to "footfall OR group OR
   demographics", since group and demographics also depend on track IDs).
2. **Staff-tag first** (`StaffFilter.mark`) — produces `exclude_ids` for this
   frame.
3. **Demographics observe** — every visible, non-excluded person gets a
   gated face-vote attempt this frame.
4. **Footfall update** — runs with `exclude_ids` so staff never count; emits
   `CrossingEvent`s.
5. For each `enter` event: resolve demographics (`.result()`), add to the
   run-level tally, and hand it to `GroupCounter.add()`.
6. An optional duck-typed `event_sink` (used by the local UI to capture
   screenshots — kept generic so the worker package never imports UI code)
   is invoked per crossing event, wrapped in try/except so a sink bug can
   never take down the detection loop.

On shutdown (`finally` in `run()`), it now logs group, staff, and
demographics summaries in addition to the footfall summary that was already
there.

**Note on a reverted change:** an annotated-debug-video renderer
(`debug_view.py`) was built and then removed during this work, after
concluding it isn't appropriate for a live/always-on feed (single
never-finalized `.mp4`, unbounded disk growth, no rotation/retention, extra
CPU competing with detection). `camera_loop.py`/`CameraSpec` currently carry
no debug-video code — this is called out here only so the history isn't a
surprise if you see it referenced elsewhere.

---

## 8. Dependencies & environment

| File | Change |
|---|---|
| `worker/requirements.txt` | `+ lap>=0.5.12` (BoT-SORT tracker association) |
| `.env.example` | `INFERENCE_FPS` 1.0→2.0, `YOLO_MODEL` yolov8n→yolo11n; added `TRACKER`, `DEMOGRAPHICS_MODEL`, `FEATURE_GROUP=true`, `FEATURE_STAFF_FILTER=true`, `STAFF_UNIFORM_HSV=[]` |
| `.gitignore` | `+ yolo11*.pt` (new model weights shouldn't be committed); `+ /ui/` (local test harness excluded from the repo) |

---

## 9. File-by-file summary

| File | Status | What changed |
|---|---|---|
| `worker/app/config.py` | modified | New settings: `tracker`, `demographics_model`, `feature_group`, `feature_staff_filter`, `staff_uniform_hsv`; defaults bumped (`inference_fps`, `yolo_model`) |
| `worker/app/models.py` | modified | `ZoneConfig.line`, `ZoneConfig.in_direction`, `polygon` now optional |
| `worker/app/zone_loader.py` | modified | Parses `line`/`in_direction` |
| `worker/app/zones.py` | modified | Added `line_to_frame_space`, `_orient`, `segments_intersect`, `crossing_is_inward` |
| `worker/app/footfall.py` | rewritten | Directional line-crossing entrance mode + polygon fallback; enter/exit/net; exclude_ids support |
| `worker/app/inference/yolo_backend.py` | modified | Tracker is configurable, not hardcoded to `bytetrack.yaml` |
| `worker/app/features/demographics.py` | rewritten | Multi-frame voting (`DemographicsAggregator`), run-level tally (`DemographicsTally`), age buckets, quality gate, configurable model pack |
| `worker/app/features/group_count.py` | new | Co-arrival group clustering off enter events |
| `worker/app/features/staff_filter.py` | new | Zone- and (optional) uniform-colour-based staff/passer-by exclusion |
| `worker/app/trackers/botsort_persist.yaml` | new | Persistence-tuned BoT-SORT profile for busy static doorways |
| `worker/app/camera_loop.py` | modified | Wires all of the above together; widened tracking condition; event_sink hook; shutdown summaries |
| `worker/app/main.py` | modified | Minor cleanup (removed a since-reverted debug-video CLI flag) |
| `worker/requirements.txt` | modified | `+ lap` |
| `.env.example`, `.gitignore` | modified | New flags/defaults; ignore new model weights and the local UI folder |
| `testdata/zones.yaml` | modified | Example line-based entrance + staff zone, with explanatory comments |

---

## 10. What this does **not** change

- No new outbound network calls / cloud dependency — InsightFace weights
  download once (from InsightFace's model zoo) and are cached locally; all
  inference is CPU-only, offline after that.
- `worker/app/features/queue_length.py` and `worker/app/features/kit_detection.py`
  are untouched — queue and kit zone types still exist in the data model and
  loader, just not exercised by any of the changes above.
- No change to `worker/app/rtsp/puller.py` (frame source / reconnect logic).
