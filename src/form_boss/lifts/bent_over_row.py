"""Bent-over row with dumbbells or a bar. Film from the side."""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Series, Signals, tempo_finding


def signals(t: PoseTrack, g: Geometry) -> Signals:
    n = g.near
    return {"elbow": G.elbow_angle(t, g, n, "sagittal"), "torso_lean": g.torso_lean,
            "upper_arm": G.arm_angle(t, g, n, "sagittal")}


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b, whole = r.bottom, slice(r.start, r.end + 1)
    return {
        "elbow_angle_top": float(sig["elbow"][b]),
        "elbow_angle_bottom": float(max(sig["elbow"][r.start], sig["elbow"][r.end])),
        "upper_arm_past_torso": float(-sig["upper_arm"][b]),
        "torso_lean": float(np.median(sig["torso_lean"][whole])),
        "torso_movement": float(np.ptp(sig["torso_lean"][whole])),
        "pulling_s": (b - r.start) / fps,
        "lowering_s": (r.end - b) / fps,
    }


def hinge(m: Metrics, g: Geometry) -> Finding:
    a = m["torso_lean"]
    if a >= 40:
        return Finding("hinge", "good", f"Torso hinged {a:.0f} deg forward.")
    return Finding("hinge", "warn" if a >= 25 else "fault",
                   f"Too upright (torso only {a:.0f} deg forward), which shifts the work to the "
                   "traps. Push your hips back until the chest is 45 deg or more toward the floor.")


def torso_still(m: Metrics, g: Geometry) -> Finding:
    # The shoulder itself travels as the shoulder blade pulls back, which reads as 10-15 deg of
    # torso movement on a strict row, so only more than that counts.
    d = m["torso_movement"]
    if d < 18:
        return Finding("torso movement", "good", "Torso angle holds steady.")
    return Finding("torso movement", "warn" if d < 30 else "fault",
                   f"Torso angle changes {d:.0f} deg during the rep; you are heaving the weight "
                   "with your back. Hold the hinge still and pull with the arms and lats.")


def pull_height(m: Metrics, g: Geometry) -> Finding:
    past, el = m["upper_arm_past_torso"], m["elbow_angle_top"]
    if past >= -5:
        return Finding("pull height", "good", "Elbows reach the torso at the top.", needs_view="side")
    return Finding("pull height", "warn" if past >= -20 else "fault",
                   f"Short pull: elbow stops {-past:.0f} deg in front of the torso (elbow bent to "
                   f"{el:.0f} deg). Drive the elbows back past your ribs.", needs_view="side")


def full_extension(m: Metrics, g: Geometry) -> Finding:
    a = m["elbow_angle_bottom"]
    if a >= 150:
        return Finding("bottom range", "good", f"Arms straighten at the bottom ({a:.0f} deg).")
    return Finding("bottom range", "warn", f"Arms stay bent at the bottom ({a:.0f} deg). Let the "
                   "weight hang to a full stretch before each pull.")


def tempo(m: Metrics, g: Geometry) -> Finding:
    return tempo_finding(m, "pulling", "lowering", 0.5)


LIFT = Lift(
    name="bent_over_row",
    view="side",
    signals=signals,
    rep_signal="elbow",
    metrics=metrics,
    checks=[hinge, torso_still, pull_height, full_extension, tempo],
    chart=[Series("elbow", "Elbow", tag="elbow"), Series("torso_lean", "Torso lean", tag="hip"),
           Series("upper_arm", "Upper arm from torso", tag="shoulder")],
    phases=("pulling", "top", "lowering"),
    min_dip=30.0,
    joints=("shoulder", "elbow", "wrist", "hip"),
    path="hand",
)
