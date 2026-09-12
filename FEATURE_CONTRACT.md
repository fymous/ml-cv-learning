# Feature contract

A capability is one file under `worker/app/features/`. Copy that file (plus a few hook lines) when you move work to another codebase.

## Must

1. **One module, one job.** `queue_length.py` counts people in a zone. It does not draw UI or write a database.
2. **Own flag.** Add `feature_<name>: bool = False` in `config.py` and `FEATURE_<NAME>` in `.env.example`.
3. **`estimate(...)` entrypoint.** Inputs come from the loop: `frame`, `detections`, `zones`, `event`, `poses`. Return a dataclass or `list[dataclass]` or `None`.
4. **Fail soft.** Catch model / crop errors inside the module. Return empty / unknown. Never raise into `camera_loop`.
5. **Lazy model load.** First call may download weights. Later calls reuse a singleton. If the flag is off, the loop must not import the heavy library (use a local import inside the `if flag` block).
6. **CPU-first.** Default providers are CPU. Do not assume a GPU.
7. **No secrets, no cloud clients, no product names** in this repo.

## Suggested signatures

Per-frame occupancy / objects:

```python
def estimate(detections, zones, frame_w, frame_h) -> list[Result]:
    ...
```

Per-event (already counted enter):

```python
def estimate(frame, event) -> Result | None:
    ...
```

Pose-based:

```python
def estimate(poses: dict[int, list[Keypoint]], detections) -> list[Result]:
    ...
```

## Hook in `camera_loop.py`

```python
if settings.feature_xyz:
    from app.features.xyz import estimate as estimate_xyz
    result = estimate_xyz(...)
    if result:
        logger.info("xyz %s", result)
```

That is the whole integration for this lab. Persistence and UI happen in another repo later.

## Backport checklist

When a feature works here:

1. Copy `worker/app/features/<name>.py`.
2. Copy the flag in config.
3. Copy the 5–10 hook lines in the camera loop.
4. Leave YAML / CLI / this README behind — the other repo already has its own wiring.

Do not rewrite the other repo’s loop to match this one. Match **this contract**, not file-for-file layout.
