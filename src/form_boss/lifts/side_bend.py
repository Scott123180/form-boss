"""Standing dumbbell side bend (oblique work). Film from the front.

The bend happens side to side, so a side view cannot see it; from there only the main fault,
folding forward instead of sideways, is visible.
"""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Profile, Series, Signals, tempo_finding


def signals(t: PoseTrack, g: Geometry) -> Signals:
    sh, hip = G.mid(t, "shoulder"), G.mid(t, "hip")
    torso = np.linalg.norm(sh - hip, axis=-1)
    if g.view == "side":
        d = t.world[:, G.JOINTS["shoulder"]].mean(1) - t.world[:, G.JOINTS["hip"]].mean(1)
        side = np.degrees(np.arctan2(d[:, 2], -d[:, 1]))          # toward/away from camera: an estimate
        forward = G.tilt(t, "hip", "shoulder", g.near) * g.facing
    else:
        d = sh - hip
        side = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))
        # Folding forward shortens the torso in the image. This is a rough estimate: a clean side
        # bend already reads 15-30 deg because the spine curves and the shoulders drop.
        forward = np.degrees(np.arccos(np.clip(torso / np.percentile(torso, 95), 0, 1)))
    side = side - np.median(side)
    shift = (hip[:, 0] - G.mid(t, "ankle")[:, 0]) / np.median(torso) * 100
    return {"side_lean": side, "forward_lean": forward,
            "hip_shift": np.abs(shift - np.median(shift)), "rep": -np.abs(side)}


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b, whole = r.bottom, slice(r.start, r.end + 1)
    return {
        "side_lean_max": float(abs(sig["side_lean"][b])),
        "forward_lean_max": float(sig["forward_lean"][whole].max()),
        "hip_shift_pct_torso": float(sig["hip_shift"][whole].max()),
        "leaning_s": (b - r.start) / fps,
        "returning_s": (r.end - b) / fps,
    }


def bend_range(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["side_lean_max"]
    if a >= p.limit(35, 20):
        return Finding("range", "good", f"Bent {a:.0f} deg to the side.", needs_view="front")
    return Finding("range", "warn", f"Shallow bend ({a:.0f} deg). Slide the weight down toward the "
                   "knee, then stand tall again.", needs_view="front")


def stays_sideways(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["forward_lean_max"]
    cue = ("which takes the load off the obliques and puts it on the lower back. Imagine bending "
           "between two panes of glass.")
    if g.view == "side":   # measured directly
        if a < p.tol(20):
            return Finding("forward fold", "good", "Bend stays side to side.")
        return Finding("forward fold", "warn" if a < p.tol(35) else "fault", f"Folding {a:.0f} deg forward, {cue}")
    if a < p.tol(45):             # estimated, so only a large fold is called out
        return Finding("forward fold", "info", "No large forward fold seen; a side view measures this properly.")
    return Finding("forward fold", "warn", f"Torso appears to fold roughly {a:.0f} deg forward, {cue}")


def hips_over_feet(m: Metrics, g: Geometry, p: Profile) -> Finding:
    s = m["hip_shift_pct_torso"]
    if s < p.tol(20):
        return Finding("hips", "good", "Hips stay over the feet.", needs_view="front")
    return Finding("hips", "warn", f"Hips push out sideways ({s:.0f}% of torso length). Keep the "
                   "hips still and bend from the waist.", needs_view="front")


def tempo(m: Metrics, g: Geometry, p: Profile) -> Finding:
    return tempo_finding(m, "leaning", "returning", 0.0)


LIFT = Lift(
    name="side_bend",
    view="front",
    signals=signals,
    rep_signal="rep",
    metrics=metrics,
    checks=[bend_range, stays_sideways, hips_over_feet, tempo],
    chart=[Series("side_lean", "Side lean", tag="shoulder"), Series("forward_lean", "Forward fold"),
           Series("hip_shift", "Hip shift", "%")],
    phases=("leaning", "end of bend", "returning"),
    min_dip=10.0,
    near=5.0,
    joints=("shoulder", "hip"),
    path="shoulder",
    path_point="shoulder",
)
