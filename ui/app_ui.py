"""Local test UI for the CV pipeline (Streamlit).

Four tabs:
  1. Zone Builder — pick an MP4, place area zones (staff/queue/kit/exclude/
     dwell/service/enroll) and the entrance tripwire line with sliders on a live
     preview, save zones.yaml.
  2. Run — choose flags and run one pass; screenshots are captured per signal.
  3. Results — summaries + galleries: people entering, groups, kids, exits.
  4. Staff Setup — learn the store's staff uniform from an `enroll` zone.

Local only — not pushed. Run with:
    cd ui && streamlit run app_ui.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml

from runner import first_frame, learn_staff_uniform, run_pipeline

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "outputs"
_OUT.mkdir(exist_ok=True)
_DEFAULT_ZONES = str(_OUT / "zones.yaml")
_DISPLAY_W = 900

_IN_DIRECTIONS = {
    "Down (into venue is downward)": [0.0, 1.0],
    "Up": [0.0, -1.0],
    "Left": [-1.0, 0.0],
    "Right": [1.0, 0.0],
    "Auto (left of the arrow)": None,
}
_AREA_TYPES = ["staff", "queue", "kit", "exclude", "dwell", "service", "enroll"]
_AREA_COLORS = {"staff": (0, 0, 255), "exclude": (0, 0, 255),
                "queue": (255, 128, 0), "kit": (0, 165, 255),
                "dwell": (200, 0, 150), "service": (255, 128, 0),
                "enroll": (0, 255, 0)}

st.set_page_config(page_title="CV Pipeline Lab", layout="wide")


def _load_display_frame(video_path: str):
    frame, w, h = first_frame(video_path)
    if frame is None:
        return None, 0, 0
    scale = _DISPLAY_W / w if w > _DISPLAY_W else 1.0
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return disp, disp.shape[1], disp.shape[0]


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------- Zone Builder
def zone_builder():
    st.header("1 · Zone Builder")
    video_path = st.text_input(
        "MP4 path", value=st.session_state.get("video_path", ""),
        placeholder="/absolute/path/to/clip.mp4",
    )
    st.session_state["video_path"] = video_path
    if not video_path or not os.path.exists(video_path):
        st.info("Enter a valid MP4 path to load the first frame.")
        return

    disp, dw, dh = _load_display_frame(video_path)
    if disp is None:
        st.error("Could not read a frame from that file.")
        return

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Entrance tripwire (directional)")
        st.caption("The count line at the door. Direction sets what 'enter' means.")
        c = st.columns(4)
        x1 = c[0].slider("x1", 0.0, 1.0, 0.30, 0.01)
        y1 = c[1].slider("y1", 0.0, 1.0, 0.75, 0.01)
        x2 = c[2].slider("x2", 0.0, 1.0, 0.70, 0.01)
        y2 = c[3].slider("y2", 0.0, 1.0, 0.75, 0.01)
        in_label = st.selectbox("Inside direction", list(_IN_DIRECTIONS.keys()))
        ent_name = st.text_input("Entrance name", value="Door")

        st.subheader("Area zones")
        st.caption(
            "Guard post = 'staff'. Product aisle / browsing section = 'dwell' "
            "(measures time spent). Sales counter / area a staff should attend = "
            "'service' (raises an alert if no staff comes for a customer). "
            "'enroll' = onboarding spot where staff stand so the app can learn "
            "their uniform (see the Staff Setup tab). Draw as boxes "
            "(fractions of the frame)."
        )
        n = st.number_input("How many area zones", 0, 8, 1)
        area = []
        for i in range(int(n)):
            st.markdown(f"**Zone {i + 1}**")
            cc = st.columns([1, 1])
            ztype = cc[0].selectbox("type", _AREA_TYPES, key=f"atype_{i}")
            zname = cc[1].text_input("name", value=ztype.capitalize(), key=f"aname_{i}")
            s = st.columns(4)
            zx = s[0].slider("x", 0.0, 1.0, 0.72, 0.01, key=f"zx_{i}")
            zy = s[1].slider("y", 0.0, 1.0, 0.55, 0.01, key=f"zy_{i}")
            zw = s[2].slider("w", 0.05, 1.0, 0.25, 0.01, key=f"zw_{i}")
            zh = s[3].slider("h", 0.05, 1.0, 0.40, 0.01, key=f"zh_{i}")
            poly = [[zx, zy], [min(1, zx + zw), zy],
                    [min(1, zx + zw), min(1, zy + zh)], [zx, min(1, zy + zh)]]
            area.append({"id": f"{ztype}-{i + 1}", "type": ztype, "name": zname, "polygon": poly})

    # ---- live preview ----
    prev = disp.copy()
    p1 = (int(x1 * dw), int(y1 * dh))
    p2 = (int(x2 * dw), int(y2 * dh))
    cv2.line(prev, p1, p2, (0, 255, 255), 3)
    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    ind = _IN_DIRECTIONS[in_label]
    dx, dy = (-(p2[1] - p1[1]), (p2[0] - p1[0])) if ind is None else ind
    nrm = (dx * dx + dy * dy) ** 0.5 or 1.0
    cv2.arrowedLine(prev, (mx, my), (int(mx + dx / nrm * 40), int(my + dy / nrm * 40)),
                    (0, 255, 255), 2, tipLength=0.4)
    for z in area:
        pts = np.array([[int(px * dw), int(py * dh)] for px, py in z["polygon"]], np.int32)
        col = _AREA_COLORS.get(z["type"], (200, 200, 200))
        cv2.polylines(prev, [pts], True, col, 2)
        cv2.putText(prev, f"{z['type']}:{z['name']}", (pts[0][0], max(12, pts[0][1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    with right:
        st.subheader("Preview")
        st.image(_rgb(prev), use_container_width=True,
                 caption="yellow = entrance line + IN arrow · red = staff · blue = queue/service · "
                         "orange = kit · purple = dwell")

    zones_path = st.text_input("Save zones.yaml to", value=_DEFAULT_ZONES)
    if st.button("💾 Save zones.yaml", type="primary"):
        entrance = {"id": "entrance-1", "type": "entrance", "name": ent_name,
                    "line": [[x1, y1], [x2, y2]]}
        if _IN_DIRECTIONS[in_label] is not None:
            entrance["in_direction"] = _IN_DIRECTIONS[in_label]
        doc = {"zones": [entrance] + area}
        Path(zones_path).parent.mkdir(parents=True, exist_ok=True)
        Path(zones_path).write_text(yaml.safe_dump(doc, sort_keys=False))
        st.session_state["zones_path"] = zones_path
        st.success(f"Saved {len(doc['zones'])} zones → {zones_path}")
        st.code(yaml.safe_dump(doc, sort_keys=False), language="yaml")


# ------------------------------------------------------------------------- Run
def run_tab():
    st.header("2 · Run pipeline (single pass)")
    video_path = st.text_input("MP4 path", value=st.session_state.get("video_path", ""))
    zones_path = st.text_input("zones.yaml", value=st.session_state.get("zones_path", _DEFAULT_ZONES))
    c = st.columns(6)
    footfall = c[0].checkbox("Footfall", True)
    group = c[1].checkbox("Groups", True)
    demographics = c[2].checkbox("Age/Gender", True)
    staff = c[3].checkbox("Staff filter", True)
    dwell = c[4].checkbox("Dwell time", True)
    attendance = c[5].checkbox("Attendance", True)
    dwell_min_sec = st.slider(
        "Dwell min seconds (ignore shorter stays)", 1.0, 30.0, 4.0, 0.5,
        disabled=not dwell,
    )
    unattended_alert_sec = st.slider(
        "Unattended alert after (s) — no staff attends a customer in a service zone",
        1.0, 30.0, 5.0, 0.5, disabled=not attendance,
    )
    if attendance and not staff:
        st.warning("Attendance needs the staff filter on to tell staff from customers.")
    fps = st.slider("Inference FPS", 1.0, 8.0, 5.0, 0.5)

    default_profile = st.session_state.get("staff_profile_path", str(_OUT / "staff_profile.json"))
    use_profile = st.checkbox(
        "Use learned staff uniform (recognises staff by uniform anywhere in frame)",
        value=os.path.exists(default_profile),
        help="Learn it first in the Staff Setup tab. Lets floor staff — not just "
             "the cashier — be recognised as staff.",
    )
    profile_path = st.text_input("Staff profile", value=default_profile, disabled=not use_profile)
    out_dir = st.text_input("Output folder", value=str(_OUT / "run"))

    if st.button("▶️ Run", type="primary"):
        if not (video_path and os.path.exists(video_path)):
            st.error("Provide a valid MP4 path.")
            return
        if not os.path.exists(zones_path):
            st.error("zones.yaml not found — build it in tab 1 first.")
            return
        with st.spinner("Running… first run downloads models (YOLO / InsightFace)."):
            summary = run_pipeline(
                video_path, zones_path, out_dir,
                fps=fps, footfall=footfall, group=group,
                demographics=demographics, staff_filter=staff, dwell=dwell,
                dwell_min_sec=dwell_min_sec, attendance=attendance,
                unattended_alert_sec=unattended_alert_sec,
                staff_profile_path=(profile_path if use_profile else None),
            )
        st.session_state["last_out_dir"] = out_dir
        st.success(f"Done — {summary['frames']} frames processed.")
        st.json(summary)


# --------------------------------------------------------------------- Results
def _dist_chart(d: dict):
    if not d:
        st.caption("— none —")
        return
    try:
        st.bar_chart(d)
    except Exception:
        st.table(list(d.items()))


def _gallery(out_dir: Path, rows: list[dict], cols: int = 4):
    if not rows:
        st.caption("— none —")
        return
    columns = st.columns(cols)
    for i, r in enumerate(rows):
        img_rel = r.get("crop") or r.get("frame")
        if not img_rel:
            continue
        img_path = out_dir / img_rel
        if not img_path.exists():
            continue
        cap = " ".join(
            f"{k}={r[k]}" for k in
            ("seq", "track_id", "gender", "age", "age_group", "size", "group_id",
             "net", "zone", "dwell_sec", "wait_sec", "party_size")
            if r.get(k) is not None
        )
        columns[i % cols].image(str(img_path), caption=cap, use_container_width=True)


def results_tab():
    st.header("3 · Results & signal screenshots")
    out_dir = Path(st.text_input(
        "Run output folder", value=st.session_state.get("last_out_dir", str(_OUT / "run"))))
    summ_path = out_dir / "summary.json"
    if not summ_path.exists():
        st.info("No results yet — run the pipeline in tab 2.")
        return
    summary = json.loads(summ_path.read_text())

    ff = summary.get("footfall", [{}])[0] if summary.get("footfall") else {}
    m = st.columns(4)
    m[0].metric("Entered", ff.get("entered", 0))
    m[1].metric("Exited", ff.get("exited", 0))
    m[2].metric("Net inside", ff.get("net", 0))
    g = summary.get("group", {})
    m[3].metric("Groups (2+)", g.get("groups_2plus", 0),
                help=f"{g.get('total_clusters', 0)} arrival clusters incl. solo")

    if summary.get("staff"):
        st.caption(f"Staff/guards excluded from the count: {summary['staff']['staff_excluded']}")

    demo = summary.get("demographics")
    if demo:
        st.subheader("Age-group / gender distribution")
        cA, cB = st.columns(2)
        with cA:
            st.caption("By age group")
            _dist_chart(demo.get("by_age_group", {}))
        with cB:
            st.caption("By gender")
            _dist_chart(demo.get("by_gender", {}))
        st.caption(f"{demo.get('with_reading', 0)} of {demo.get('counted', 0)} visitors got a face reading.")

    dwell = summary.get("dwell")
    if dwell:
        st.subheader("🕐 Dwell time (time spent per zone)")
        st.caption(
            f"Stays shorter than {summary.get('dwell_min_sec', 4.0)}s are ignored "
            "(not highlighted, not counted)."
        )
        for zd in dwell:
            st.markdown(f"**{zd['zone_name']}** (`{zd['zone_id']}`)")
            dcols = st.columns(5)
            dcols[0].metric("Visits", zd["visits"])
            dcols[1].metric("Avg dwell", f"{zd['avg_dwell_sec']}s")
            dcols[2].metric("Median dwell", f"{zd['median_dwell_sec']}s")
            dcols[3].metric("Max dwell", f"{zd['max_dwell_sec']}s")
            dcols[4].metric("Currently inside", zd["currently_inside"])
            _dist_chart(zd.get("bucket_histogram", {}))

    attend = summary.get("attendance")
    if attend:
        st.subheader("🧑‍💼 Customer attendance (service zones)")
        st.caption(
            f"A customer in a service zone with no staff coming within reach for "
            f"{summary.get('unattended_alert_sec', 5.0)}s raises one 'unattended' alert "
            "(per group). Once a staff attends them they're marked attended and never "
            "re-alerted. Needs the staff filter on to identify staff."
        )
        for za in attend:
            st.markdown(f"**{za['zone_name']}** (`{za['zone_id']}`)")
            acols = st.columns(4)
            acols[0].metric("Customers seen", za["customers_seen"])
            acols[1].metric("Attended", za["attended"])
            acols[2].metric("Unattended alerts", za["unattended_alerts"])
            acols[3].metric("Currently waiting", za["currently_waiting"])

    events = []
    jl = Path(summary.get("events_jsonl", out_dir / "events.jsonl"))
    if jl.exists():
        events = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]

    def of(kind):
        return [e for e in events if e.get("kind") == kind]

    st.subheader("👤 People entering (count screenshots)")
    _gallery(out_dir, of("enter"))
    st.subheader("👨‍👩‍👧 Groups (2+ people arriving together)")
    _gallery(out_dir, of("group"))
    st.subheader("🧒 Kids detected")
    _gallery(out_dir, of("kid"))
    st.subheader("🚪 Exits")
    _gallery(out_dir, of("exit"))
    st.subheader("🕐 Dwell events (zone left / finished browsing)")
    st.caption("Snapshot taken the moment the visit ends — the departed person "
               "may no longer be visible in-frame; zone outline shown for context.")
    _gallery(out_dir, of("dwell"))
    st.subheader("🚨 Unattended customers (no staff attended)")
    _gallery(out_dir, of("unattended"))
    st.subheader("✅ Attended customers (staff came)")
    _gallery(out_dir, of("attended"))


# -------------------------------------------------------------- Staff Setup
def _swatch(bgr, size: int = 64):
    """Solid RGB swatch image for a BGR colour tuple."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # to RGB
    return img


def staff_setup_tab():
    st.header("4 · Staff Setup (learn the uniform)")
    st.caption(
        "Onboarding: in the Zone Builder, draw an **enroll** zone over a spot "
        "where staff will stand. Ask staff (one or a few) to stand in it and "
        "slowly turn around for ~10–15s. Then learn the uniform here — the app "
        "samples their clothing colour and saves a staff profile. After that, "
        "staff are recognised by uniform **anywhere in frame** (floor staff, "
        "not just the cashier), which powers attendance and cleaner footfall."
    )
    video_path = st.text_input("MP4 path", value=st.session_state.get("video_path", ""),
                               key="staff_video")
    zones_path = st.text_input("zones.yaml (must contain an enroll zone)",
                               value=st.session_state.get("zones_path", _DEFAULT_ZONES),
                               key="staff_zones")
    profile_path = st.text_input("Save staff profile to",
                                 value=str(_OUT / "staff_profile.json"))
    c = st.columns(2)
    seconds = c[0].slider("Capture seconds", 3.0, 60.0, 15.0, 1.0)
    fps = c[1].slider("Sample FPS", 1.0, 8.0, 4.0, 0.5)

    if st.button("🎓 Learn staff uniform", type="primary"):
        if not (video_path and os.path.exists(video_path)):
            st.error("Provide a valid MP4 path.")
            return
        if not os.path.exists(zones_path):
            st.error("zones.yaml not found — build it (with an enroll zone) in tab 1.")
            return
        try:
            with st.spinner("Learning uniform from people in the enroll zone…"):
                res = learn_staff_uniform(video_path, zones_path, profile_path,
                                          fps=fps, seconds=seconds, yolo_model="yolo11n.pt")
        except ValueError as e:
            st.error(str(e))
            return
        st.session_state["staff_profile_path"] = res["profile_path"]
        if res["sample_count"] == 0:
            st.warning(
                "No samples collected — nobody's feet were inside the enroll zone "
                "during the capture window. Check the zone position and that staff "
                "stand in it."
            )
        else:
            st.success(
                f"Learned {len(res['uniform_hsv'])} uniform colour range(s) from "
                f"{res['sample_count']} samples across {res['tracks_sampled']} people. "
                f"Saved → {res['profile_path']}"
            )
        if res.get("warning"):
            st.warning(res["warning"])
        if res["swatches"]:
            st.caption("Learned uniform colour(s):")
            scols = st.columns(min(6, len(res["swatches"])))
            for i, bgr in enumerate(res["swatches"]):
                scols[i % len(scols)].image(_swatch(bgr), caption=str(res["uniform_hsv"][i]))
        if res.get("preview") and os.path.exists(res["preview"]):
            st.image(res["preview"], caption="Green box = sampled staff · green outline = enroll zone",
                     use_container_width=True)
        st.info("Now go to the **Run** tab — 'Use learned staff uniform' will be on.")


tab1, tab2, tab3, tab4 = st.tabs(["Zone Builder", "Run", "Results", "Staff Setup"])
with tab1:
    zone_builder()
with tab2:
    run_tab()
with tab3:
    results_tab()
with tab4:
    staff_setup_tab()
