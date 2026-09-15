"""EventCapture — a camera_loop event sink that saves per-signal screenshots.

Implements the duck-typed sink `camera_loop` calls:
    on_crossing(frame, detections, exclude_ids, ev, demo, group_result,
                footfall_summary, media_ts)
    on_dwell(frame, detections, exclude_ids, dwell_event, media_ts)
    on_attendance(frame, detections, exclude_ids, attendance_event, media_ts)

For every counted signal it writes an annotated full-frame JPG (+ a tight crop
for crossings) and appends a row to `events.jsonl`. The Streamlit UI reads
that folder to show galleries: people entering, exits, groups forming,
kids/demographics, dwell (time-in-zone), and attendance (attended /
unattended-customer) events.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

_C_PERSON = (0, 220, 0)
_C_EXCLUDED = (0, 0, 255)
_C_EVENT = (0, 255, 255)
_C_GROUP = (255, 0, 255)  # magenta — members of a 2+ co-arrival group
_C_DWELL_ZONE = (200, 0, 150)  # purple — dwell zone outline
_C_SERVICE_ZONE = (255, 128, 0)  # blue-ish — service (attendance) zone outline
_C_ALERT = (0, 0, 255)  # red — unattended customer
_C_ATTENDED = (0, 200, 0)  # green — attended customer


@dataclass
class _Dirs:
    root: str
    enter: str
    exit: str
    group: str
    kid: str
    dwell: str
    attend: str

    @classmethod
    def make(cls, root: str) -> "_Dirs":
        d = cls(
            root=root,
            enter=os.path.join(root, "enter"),
            exit=os.path.join(root, "exit"),
            group=os.path.join(root, "group"),
            kid=os.path.join(root, "kid"),
            dwell=os.path.join(root, "dwell"),
            attend=os.path.join(root, "attend"),
        )
        for p in (d.root, d.enter, d.exit, d.group, d.kid, d.dwell, d.attend):
            os.makedirs(p, exist_ok=True)
        return d


class EventCapture:
    def __init__(self, out_dir: str, zones: list | None = None) -> None:
        self._d = _Dirs.make(out_dir)
        self._jsonl = os.path.join(out_dir, "events.jsonl")
        # start fresh each run
        open(self._jsonl, "w").close()
        self._last_group_size: dict[int, int] = {}
        self._dwell_zones = [z for z in (zones or []) if getattr(z, "zone_type", None) == "dwell"]
        self._dwell_seq = 0
        self._service_zones = [z for z in (zones or []) if getattr(z, "zone_type", None) == "service"]
        self._attend_seq = 0

    # ------------------------------------------------------------------ sink
    def on_crossing(
        self, frame, detections, exclude_ids, ev, demo, group_result,
        footfall_summary, media_ts,
    ) -> None:
        seq = ev.seq_n
        base = self._annotated(frame, detections, exclude_ids, ev, demo, group_result, media_ts)
        crop = self._crop(frame, ev)

        # zone-level running totals for the panel
        zsum = next((z for z in footfall_summary if z["zone_id"] == ev.zone_id), {})

        if ev.direction == "enter":
            fpath = os.path.join(self._d.enter, f"enter_{seq:04d}.jpg")
            cpath = os.path.join(self._d.enter, f"enter_{seq:04d}_crop.jpg")
            cv2.imwrite(fpath, base)
            if crop is not None:
                cv2.imwrite(cpath, crop)
            self._log({
                "kind": "enter", "ts": round(media_ts, 2), "track_id": ev.track_id,
                "seq": seq, "zone": ev.zone_name,
                "entered": zsum.get("entered"), "net": zsum.get("net"),
                "gender": getattr(demo, "gender", None),
                "age": round(demo.age_estimate, 1) if getattr(demo, "age_estimate", None) is not None else None,
                "age_group": getattr(demo, "age_group", None),
                "votes": getattr(demo, "votes", None),
                "frame": os.path.relpath(fpath, self._d.root),
                "crop": os.path.relpath(cpath, self._d.root) if crop is not None else None,
            })
            # kid gallery
            if getattr(demo, "age_group", None) == "0-12":
                kpath = os.path.join(self._d.kid, f"kid_{seq:04d}.jpg")
                cv2.imwrite(kpath, crop if crop is not None else base)
                self._log({
                    "kind": "kid", "ts": round(media_ts, 2), "track_id": ev.track_id,
                    "seq": seq, "age": round(demo.age_estimate, 1) if demo.age_estimate else None,
                    "age_group": demo.age_group, "gender": demo.gender,
                    "frame": os.path.relpath(kpath, self._d.root),
                })
            # group gallery — ONLY genuine multi-person co-arrivals (size >= 2),
            # captured once each time the group grows. Solo arrivals are not
            # "groups" and belong only in the enter gallery.
            if group_result is not None and group_result.size >= 2:
                prev = self._last_group_size.get(group_result.group_id, 0)
                if group_result.size != prev:
                    self._last_group_size[group_result.group_id] = group_result.size
                    gimg = self._annotated(
                        frame, detections, exclude_ids, ev, None, group_result,
                        media_ts, group_members=set(group_result.member_track_ids),
                    )
                    gpath = os.path.join(
                        self._d.group,
                        f"group_{group_result.group_id:03d}_size{group_result.size}.jpg",
                    )
                    cv2.imwrite(gpath, gimg)
                    self._log({
                        "kind": "group", "ts": round(media_ts, 2),
                        "group_id": group_result.group_id, "size": group_result.size,
                        "members": group_result.member_track_ids,
                        "frame": os.path.relpath(gpath, self._d.root),
                    })
        else:  # exit
            fpath = os.path.join(self._d.exit, f"exit_{seq:04d}.jpg")
            cv2.imwrite(fpath, base)
            self._log({
                "kind": "exit", "ts": round(media_ts, 2), "track_id": ev.track_id,
                "seq": seq, "zone": ev.zone_name,
                "exited": zsum.get("exited"), "net": zsum.get("net"),
                "frame": os.path.relpath(fpath, self._d.root),
            })

    # -------------------------------------------------------------- dwell
    def on_dwell(self, frame, detections, exclude_ids, dev, media_ts) -> None:
        self._dwell_seq += 1
        seq = self._dwell_seq
        caption = (
            f"DWELL zone={dev.zone_name} track={dev.track_id} "
            f"{dev.dwell_sec:.1f}s t={media_ts:.1f}s"
        )
        img = self._annotated_plain(frame, detections, exclude_ids, caption)
        fpath = os.path.join(self._d.dwell, f"dwell_{seq:04d}.jpg")
        cv2.imwrite(fpath, img)
        self._log({
            "kind": "dwell", "ts": round(media_ts, 2), "track_id": dev.track_id,
            "seq": seq, "zone": dev.zone_name, "zone_id": dev.zone_id,
            "dwell_sec": round(dev.dwell_sec, 1),
            "frame": os.path.relpath(fpath, self._d.root),
        })

    # ----------------------------------------------------------- attendance
    def on_attendance(self, frame, detections, exclude_ids, aev, media_ts) -> None:
        self._attend_seq += 1
        seq = self._attend_seq
        if aev.kind == "unattended":
            caption = (
                f"UNATTENDED {aev.wait_sec:.1f}s  zone={aev.zone_name} "
                f"customer={aev.track_id} party={aev.party_size} t={media_ts:.1f}s"
            )
            hl = _C_ALERT
        else:  # attended
            caption = (
                f"ATTENDED after {aev.wait_sec:.1f}s  zone={aev.zone_name} "
                f"customer={aev.track_id} t={media_ts:.1f}s"
            )
            hl = _C_ATTENDED
        img = self._annotated_service(frame, detections, exclude_ids, aev, hl, caption)
        fpath = os.path.join(self._d.attend, f"{aev.kind}_{seq:04d}.jpg")
        cv2.imwrite(fpath, img)
        self._log({
            "kind": aev.kind, "ts": round(media_ts, 2), "track_id": aev.track_id,
            "seq": seq, "zone": aev.zone_name, "zone_id": aev.zone_id,
            "wait_sec": round(aev.wait_sec, 1), "party_size": aev.party_size,
            "frame": os.path.relpath(fpath, self._d.root),
        })

    def _annotated_service(self, frame, detections, exclude_ids, aev, hl, caption: str):
        """Draw service zones + staff (red) / customers (green), and highlight
        the event customer's bbox in the alert/attended colour."""
        img = frame.copy()
        h, w = img.shape[:2]
        for z in self._service_zones:
            poly = getattr(z, "polygon", None)
            if not poly:
                continue
            pts = np.array([[int(px * w), int(py * h)] for px, py in poly], np.int32)
            cv2.polylines(img, [pts], True, _C_SERVICE_ZONE, 2)
            cv2.putText(img, z.name or "service", (pts[0][0], max(12, pts[0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_SERVICE_ZONE, 1, cv2.LINE_AA)
        for d in detections:
            if d.class_name != "person":
                continue
            staff = d.track_id is not None and d.track_id in exclude_ids
            color = _C_EXCLUDED if staff else _C_PERSON
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
            lbl = f"ID{d.track_id}" + (" STAFF" if staff else "")
            cv2.putText(img, lbl, (int(d.x1), max(12, int(d.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # highlight the event customer
        cv2.rectangle(img, (int(aev.x1), int(aev.y1)), (int(aev.x2), int(aev.y2)), hl, 4)
        cv2.rectangle(img, (0, 0), (w, 26), (20, 20, 20), -1)
        cv2.putText(img, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _annotated_plain(self, frame, detections, exclude_ids, caption: str):
        """Generic annotator for events with no single bbox to highlight
        (e.g. dwell — the track has already left by the time it finalizes)."""
        img = frame.copy()
        h, w = img.shape[:2]
        for z in self._dwell_zones:
            poly = getattr(z, "polygon", None)
            if not poly:
                continue
            pts = np.array([[int(px * w), int(py * h)] for px, py in poly], np.int32)
            cv2.polylines(img, [pts], True, _C_DWELL_ZONE, 2)
            cv2.putText(img, z.name or "dwell", (pts[0][0], max(12, pts[0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_DWELL_ZONE, 1, cv2.LINE_AA)
        for d in detections:
            if d.class_name != "person":
                continue
            excluded = d.track_id is not None and d.track_id in exclude_ids
            color = _C_EXCLUDED if excluded else _C_PERSON
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
            lbl = f"ID{d.track_id}" + (" STAFF" if excluded else "")
            cv2.putText(img, lbl, (int(d.x1), max(12, int(d.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w, 26), (20, 20, 20), -1)
        cv2.putText(img, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # --------------------------------------------------------------- drawing
    def _annotated(self, frame, detections, exclude_ids, ev, demo, group_result,
                   media_ts, group_members=None):
        img = frame.copy()
        members = group_members or set()
        for d in detections:
            if d.class_name != "person":
                continue
            excluded = d.track_id is not None and d.track_id in exclude_ids
            in_group = d.track_id is not None and d.track_id in members
            if excluded:
                color, tag = _C_EXCLUDED, " STAFF"
            elif in_group:
                color, tag = _C_GROUP, " GROUP"
            else:
                color, tag = _C_PERSON, ""
            thick = 3 if in_group else 2
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, thick)
            lbl = f"ID{d.track_id}{tag}"
            cv2.putText(img, lbl, (int(d.x1), max(12, int(d.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # highlight the event person
        cv2.rectangle(img, (int(ev.x1), int(ev.y1)), (int(ev.x2), int(ev.y2)), _C_EVENT, 3)
        caption = f"{ev.direction.upper()} #{ev.seq_n} t={media_ts:.1f}s"
        if demo is not None and (demo.gender or demo.age_group):
            caption += f" | {demo.gender or '?'} {demo.age_group or '?'}"
        if group_result is not None:
            caption += f" | group {group_result.group_id} size {group_result.size}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(img, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _crop(self, frame, ev):
        h, w = frame.shape[:2]
        bw, bh = ev.x2 - ev.x1, ev.y2 - ev.y1
        x1 = max(0, int(ev.x1 - 0.15 * bw))
        y1 = max(0, int(ev.y1 - 0.15 * bh))
        x2 = min(w, int(ev.x2 + 0.15 * bw))
        y2 = min(h, int(ev.y2 + 0.15 * bh))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _log(self, row: dict) -> None:
        with open(self._jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
