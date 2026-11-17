"""Annotated output video: skeleton, bar path, live angles, per-rep verdict."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import geometry
from . import pose as P
from .lifts.base import Lift, RepReport, Signals
from .pose import PoseTrack
from .reps import Rep

COLORS = {"good": (90, 200, 90), "warn": (0, 190, 255), "fault": (60, 60, 235), "info": (220, 220, 220)}


def _text(img, s, org, scale, color=(255, 255, 255)):
    th = 1 if scale < 0.9 else 2
    (tw, tht), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    box = img[max(0, y - tht - 6):y + base + 4, max(0, x - 6):x + tw + 6]
    box[:] = (box * 0.35).astype(box.dtype)  # dark backing so text reads on any footage
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def render(video: Path, out: Path, t: PoseTrack, lift: Lift, sig: Signals, reps: list[Rep],
           reports: list[RepReport]) -> Path:
    path = geometry.mid(t, lift.path_point)
    cap = cv2.VideoCapture(str(video))
    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), t.fps, (t.width, t.height))
    k = max(0.75, min(t.width, t.height) / 720)  # scale drawing to resolution
    lw = max(2, int(3 * k))

    for i in range(len(t)):
        ok, frame = cap.read()
        if not ok:
            break
        pts = t.image[i].astype(int)

        for a, b in P.SKELETON:
            if min(t.vis[i, a], t.vis[i, b]) > 0.3:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 200, 60), lw, cv2.LINE_AA)
        for j in range(11, 33):
            if t.vis[i, j] > 0.3:
                cv2.circle(frame, tuple(pts[j]), lw + 1, (255, 255, 255), -1, cv2.LINE_AA)

        # Path of the bar/hands for the current rep (or the last 1.5 s between reps)
        cur = next((n for n, r in enumerate(reps) if r.start <= i <= r.end), None)
        s = reps[cur].start if cur is not None else max(0, i - int(1.5 * t.fps))
        trail = path[s:i + 1].astype(np.int32)
        if len(trail) > 1:
            cv2.polylines(frame, [trail], False, (60, 60, 235), lw, cv2.LINE_AA)
        cv2.circle(frame, tuple(path[i].astype(int)), lw + 3, (60, 60, 235), -1, cv2.LINE_AA)

        done = sum(1 for r in reps if r.bottom <= i)
        y = int(44 * k)
        _text(frame, f"Rep {done}/{len(reps)}", (int(16 * k), y), 1.0 * k)
        for s in lift.chart:
            y += int(34 * k)
            _text(frame, f"{s.label.lower()} {sig[s.key][i]:4.0f}", (int(16 * k), y), 0.7 * k)

        # Show the verdict from each rep's bottom until the next rep starts
        shown = next((n for n in reversed(range(len(reps))) if reps[n].bottom <= i), None)
        if shown is not None and (shown + 1 >= len(reps) or i < reps[shown + 1].start):
            yy = t.height - int(20 * k) - int(34 * k) * len(reports[shown].findings)
            for f in reports[shown].findings:
                yy += int(34 * k)
                _text(frame, f"{f.check}: {f.message.split('. ')[0].split(';')[0].rstrip('.')[:60]}",
                      (int(16 * k), yy), 0.6 * k, COLORS[f.status])
        writer.write(frame)

    cap.release()
    writer.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
        tmp.unlink()
    else:
        tmp.rename(out)
    return out
