"""Push-up. Film from the side, camera low and level with the body."""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Series, Signals, tempo_finding


def signals(t: PoseTrack, g: Geometry) -> Signals:
    n = g.near
    x = t.image if g.view == "side" else t.world
    sh, hip, ank = (x[:, G.JOINTS[j][n]] for j in ("shoulder", "hip", "ankle"))
    # Hip position relative to the straight shoulder-ankle line: + = sagging below it, - = piked above.
    along = ((hip - sh) * (ank - sh)).sum(-1) / ((ank - sh) ** 2).sum(-1)
    below = hip[:, 1] > sh[:, 1] + along * (ank[:, 1] - sh[:, 1])   # y points down
    hip_line = (180 - G.joint_angle(t, g, "shoulder", "hip", "ankle", n, "sagittal")) * np.where(below, 1, -1)
    # How far the shoulder sits above the hand, as % of arm length: ~100 at lockout, near 0 on the floor.
    return {"elbow": G.elbow_angle(t, g, n, "sagittal"), "hip_line": hip_line,
            "shoulder_height": -G.hand_height(t, n)}


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b, whole = r.bottom, slice(r.start, r.end + 1)
    return {
        "elbow_angle_bottom": float(sig["elbow"][b]),
        "elbow_angle_top": float(max(sig["elbow"][r.start], sig["elbow"][r.end])),
        "shoulder_height_bottom_pct": float(sig["shoulder_height"][b]),
        "hip_sag_max": float(sig["hip_line"][whole].max()),
        "hip_pike_max": float(-sig["hip_line"][whole].min()),
        "descent_s": (b - r.start) / fps,
        "ascent_s": (r.end - b) / fps,
    }


def depth(m: Metrics, g: Geometry) -> Finding:
    a = m["elbow_angle_bottom"]
    if a <= 95:
        return Finding("depth", "good", f"Full depth (elbows bent to {a:.0f} deg).")
    if a <= 110:
        return Finding("depth", "warn", f"Slightly short (elbows bent to {a:.0f} deg). Lower until "
                       "your upper arms are at least parallel to the floor.")
    return Finding("depth", "fault", f"Half rep (elbows only bent to {a:.0f} deg). Lower your chest "
                   "to a fist's height off the floor.")


def lockout(m: Metrics, g: Geometry) -> Finding:
    a = m["elbow_angle_top"]
    if a >= 155:
        return Finding("lockout", "good", f"Arms straighten at the top ({a:.0f} deg).")
    return Finding("lockout", "warn", f"Arms stay bent at the top ({a:.0f} deg). Press all the way "
                   "up and finish each rep with straight arms.")


def body_line(m: Metrics, g: Geometry) -> Finding:
    sag, pike = m["hip_sag_max"], m["hip_pike_max"]
    if sag >= 15:
        return Finding("body line", "fault" if sag >= 25 else "warn",
                       f"Hips sag {sag:.0f} deg below the shoulder-ankle line. Squeeze glutes and "
                       "brace your abs so the body moves as one plank.", needs_view="side")
    if pike >= 15:
        return Finding("body line", "fault" if pike >= 25 else "warn",
                       f"Hips piked {pike:.0f} deg above the shoulder-ankle line. Lower your hips "
                       "until shoulders, hips and ankles line up.", needs_view="side")
    return Finding("body line", "good", "Shoulders, hips and ankles stay in line.", needs_view="side")


def tempo(m: Metrics, g: Geometry) -> Finding:
    return tempo_finding(m, "descent", "ascent", 0.4)


LIFT = Lift(
    name="push_up",
    view="side",
    signals=signals,
    rep_signal="elbow",
    metrics=metrics,
    checks=[depth, lockout, body_line, tempo],
    chart=[Series("elbow", "Elbow", tag="elbow"), Series("hip_line", "Hip sag", tag="hip"),
           Series("shoulder_height", "Shoulder height", "%")],
    phases=("descent", "bottom", "press"),
    min_dip=30.0,
    joints=("shoulder", "elbow", "wrist", "hip", "ankle"),
    path="shoulder",
    path_point="shoulder",
)
