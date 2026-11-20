"""Standing dumbbell or barbell curl, both arms together or alternating. Film from the side."""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Profile, Series, Signals, tempo_finding


def signals(t: PoseTrack, g: Geometry) -> Signals:
    el = [G.elbow_angle(t, g, s, "sagittal") for s in (0, 1)]
    arm = [G.arm_angle(t, g, s, "sagittal") for s in (0, 1)]
    working = el[1] < el[0]   # per frame: the arm that is more bent is the one curling
    lean = (G.tilt(t, "hip", "shoulder", g.near) * g.facing if g.view == "side"
            else np.abs(G.tilt_mid(t, "hip", "shoulder")))   # + = leaning forward, from the side
    return {"elbow_l": el[0], "elbow_r": el[1], "arm_l": arm[0], "arm_r": arm[1],
            "elbow": np.minimum(*el), "upper_arm": np.where(working, arm[1], arm[0]), "torso_lean": lean}


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b = r.bottom
    s = "r" if sig["elbow_r"][b] < sig["elbow_l"][b] else "l"
    el, arm = sig["elbow_" + s], sig["arm_" + s]
    return {
        "elbow_angle_top": float(el[b]),
        "elbow_angle_bottom": float(max(el[r.start], el[r.end])),
        "upper_arm_swing": float(arm[r.start:b + 1].max() - arm[r.start]),
        "torso_swing": float(np.ptp(sig["torso_lean"][r.start:r.end + 1])),
        "right_arm": float(s == "r"),
        "lifting_s": (b - r.start) / fps,
        "lowering_s": (r.end - b) / fps,
    }


def top_range(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["elbow_angle_top"]
    if a <= p.limit(40, 60):
        return Finding("top range", "good", f"Curled all the way up (elbow closed to {a:.0f} deg).",
                       needs_view="side")
    if a <= p.limit(40, 80):
        return Finding("top range", "warn", f"Stopped a little short (elbow closed to {a:.0f} deg). "
                       "Bring the weight up to shoulder height and squeeze.", needs_view="side")
    return Finding("top range", "fault", f"Half rep (elbow only closed to {a:.0f} deg). Curl until "
                   "the forearm meets the biceps; use less weight if you cannot.", needs_view="side")


def full_extension(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["elbow_angle_bottom"]
    if a >= p.limit(180, 150):
        return Finding("bottom range", "good", f"Arm straightens at the bottom ({a:.0f} deg).")
    return Finding("bottom range", "warn", f"Arm stays bent at the bottom ({a:.0f} deg). Lower "
                   "until the arm is straight before the next rep.")


def elbow_drift(m: Metrics, g: Geometry, p: Profile) -> Finding:
    d = m["upper_arm_swing"]
    if d < p.tol(25):
        return Finding("elbow position", "good", "Upper arm stays by your side.", needs_view="side")
    return Finding("elbow position", "warn" if d < p.tol(40) else "fault",
                   f"Upper arm swings {d:.0f} deg forward as you lift, so the shoulder is doing the "
                   "work. Pin your elbow to your ribs and move only the forearm.", needs_view="side")


def body_swing(m: Metrics, g: Geometry, p: Profile) -> Finding:
    d = m["torso_swing"]
    if d < p.tol(10):
        return Finding("body swing", "good", "Torso stays still.", needs_view="side")
    return Finding("body swing", "warn" if d < p.tol(18) else "fault",
                   f"Torso rocks {d:.0f} deg during the rep; you are swinging the weight up. Stand "
                   "tall, brace, and drop the weight if you need momentum.", needs_view="side")


def tempo(m: Metrics, g: Geometry, p: Profile) -> Finding:
    return tempo_finding(m, "lifting", "lowering", 0.6)


LIFT = Lift(
    name="bicep_curl",
    view="side",
    signals=signals,
    rep_signal="elbow",
    metrics=metrics,
    checks=[top_range, full_extension, elbow_drift, body_swing, tempo],
    chart=[Series("elbow", "Elbow", tag="elbow"), Series("upper_arm", "Upper arm from torso", tag="shoulder"),
           Series("torso_lean", "Torso lean")],
    phases=("lifting", "top", "lowering"),
    joints=("shoulder", "elbow", "wrist"),
    path="hand",
)
