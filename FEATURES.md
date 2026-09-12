# Features to develop here

Build these in this repo, **one feature per Cursor chat**. Start at the top.

On the laptop: paste `PROMPT.md` as the first message, then add one line such as:

`The feature I want is: queue wait estimate.`

---

## Do first (high value, easy to copy later)

1. **Queue wait estimate** — not just “how many in the zone now,” but time-in-zone per track → estimated wait. Headcount already exists in `features/queue_length.py`; wait time is the useful KPI.

2. **Enter + exit + net occupancy** — footfall today is enter-only. Add exit so `net = in − out`. Needed for room / space occupancy, not just arrivals.

3. **Re-entry debounce / ReID-lite** — same door, exit then enter within ~20 min = same person, don’t increment. Start with a time window; add appearance embedding later.

4. **Loitering** — same `track_id` inside a zone longer than T seconds. Vacant rooms, keep-out areas, after a space should be empty.

5. **Staffed counter** — two zones: customer side + staff side. Flag when a customer is present and staff is absent.

6. **Restricted-area dwell** — person in a keep-out zone for N seconds (not one noisy frame).

---

## Do next (pose / objects)

7. **Kit / object-in-zone** — replace the COCO stub in `kit_detection.py`: bags, tools, PPE (helmet, vest). Needs a better model if COCO is not enough.

8. **Fall / lying down** — `FEATURE_POSE` + horizontal skeleton, sustained N frames.

9. **Seated vs standing** — counter / lounge occupancy quality.

10. **Phone-to-ear / distracted** — wrist near head. Optional; weaker use case.

11. **Group / together** — tracks that enter close in space and time (group vs solo).

---

## Do later (harder, still lab-shaped)

12. **Appearance ReID** — embedding gallery per door. Real same-person matching.

13. **After-hours motion** — person detected outside a local time window.

14. **Line of interest (in/out vector)** — crossing a directed line, not just “feet in polygon.” More accurate at wide doors.

15. **Crowd density / heatmap** — accumulate feet-points, write a local PNG at end of run.

16. **PPE compliance** — helmet / vest missing in a zone. Needs a dedicated model.

---

## Skip in this repo

Do **not** add: cloud deploy, auth, dashboards, databases, push notifications.

Do **not** rewrite from scratch: the YOLO + ByteTrack loop, entrance enter-count, demographics, simple queue headcount. Improve them; don’t replace them.

---

## Suggested order (first two weeks)

| Week | Features |
|---|---|
| 1 | Queue wait, enter+exit net occupancy, loitering |
| 2 | Staffed counter, restricted dwell, re-entry debounce |

Those six cover retail floors and occupancy-style alerts (mismatch vs expected count, vacant space, after a space should be empty, restricted area) without a new model download.

---

## After a feature works

Follow `FEATURE_CONTRACT.md`. Copy only:

1. `worker/app/features/<name>.py`
2. the flag in `config.py`
3. the hook lines in `camera_loop.py`
