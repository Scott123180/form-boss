"""Standing overhead triceps extension (one weight held in both hands behind the head).

Film from the front: that shows elbow flare and range of motion. A side view adds upper-arm
position and back arch, but the arms wrap around the head from there and tracking often fails.
"""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Series, Signals, tempo_finding


def signals(t: PoseTrack, g: Geometry) -> Signals:
    sides = (g.near,) if g.view == "side" else (0, 1)
    elbow = np.mean([G.elbow_angle(t, g, s, "sagittal") for s in sides], axis=0)
    # Upper arm from vertical, + = elbow drifting forward of the shoulder (side view).
    upper = (G.tilt(t, "shoulder", "elbow", g.near) * g.facing if g.view == "side"
             else 180 - np.mean([G.arm_angle(t, g, s, "sagittal") for s in (0, 1)], axis=0))
    # Elbow width as a multiple of shoulder width (front view).
    im = t.image
    width = lambda j: np.abs(im[:, G.JOINTS[j][0], 0] - im[:, G.JOINTS[j][1], 0])
    flare = width("elbow") / np.median(width("shoulder"))
    lean = (-G.tilt(t, "hip", "shoulder", g.near) * g.facing if g.view == "side"
            else np.abs(G.tilt_mid(t, "hip", "shoulder")))   # + = leaning back, from the side
    hand = np.mean([G.hand_height(t, s) for s in sides], axis=0)
    return {"elbow": elbow, "upper_arm": upper, "flare": flare, "torso_lean": lean, "hand": hand}


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b, whole = r.bottom, slice(r.start, r.end + 1)
    return {
        "elbow_angle_bottom": float(sig["elbow"][b]),
        "elbow_angle_top": float(max(sig["elbow"][r.start], sig["elbow"][r.end])),
        "upper_arm_from_vertical": float(np.median(sig["upper_arm"][whole])),
        "upper_arm_travel": float(np.ptp(sig["upper_arm"][whole])),
        "elbow_width_x_shoulders": float(sig["flare"][whole].max()),
        "hand_height_top_pct": float(max(sig["hand"][r.start], sig["hand"][r.end])),
        "hand_height_bottom_pct": float(sig["hand"][b]),
        "torso_lean_max": float(sig["torso_lean"][whole].max()),
        "lowering_s": (b - r.start) / fps,
        "extending_s": (r.end - b) / fps,
    }


# Range of motion is read from the elbow angle when filmed from the side. From the front the forearms
# fold away from the camera, so hand height is used instead: about a third of an arm length above
# the shoulders with the weight behind the head, a full arm length at lockout.

def stretch(m: Metrics, g: Geometry) -> Finding:
    a, h = m["elbow_angle_bottom"], m["hand_height_bottom_pct"]
    cue = "Lower the weight behind your head until the forearms pass parallel to the floor."
    if g.view == "side":
        if a <= 75:
            return Finding("bottom range", "good", f"Full stretch behind the head (elbows bent to {a:.0f} deg).")
        return Finding("bottom range", "warn" if a <= 95 else "fault",
                       f"Short at the bottom (elbows only bent to {a:.0f} deg). {cue}")
    if h <= 60:   # the hands are hidden behind the head here, so only a clearly short rep is called out
        return Finding("bottom range", "good", "Weight comes down behind the head.")
    return Finding("bottom range", "warn",
                   f"Short at the bottom (hands stay {h:.0f}% of arm length above the shoulders). {cue}")


def lockout(m: Metrics, g: Geometry) -> Finding:
    a, h = m["elbow_angle_top"], m["hand_height_top_pct"]
    if (a >= 155) if g.view == "side" else (h >= 80):
        return Finding("lockout", "good", "Arms straighten overhead.")
    how = f"elbows reach {a:.0f} deg" if g.view == "side" else f"hands reach {h:.0f}% of full height"
    return Finding("lockout", "warn", f"Arms stay bent at the top ({how}). Extend until the elbows "
                   "are straight.")


def upper_arm(m: Metrics, g: Geometry) -> Finding:
    tilt, travel = m["upper_arm_from_vertical"], m["upper_arm_travel"]
    if travel >= 25:
        return Finding("upper arm", "warn" if travel < 40 else "fault",
                       f"Upper arms move {travel:.0f} deg during the rep, turning it into a press. "
                       "Keep the elbows fixed in space and hinge only at the elbow.", needs_view="side")
    if tilt >= 30:
        return Finding("upper arm", "warn", f"Elbows sit {tilt:.0f} deg forward of vertical. Point "
                       "them at the ceiling, biceps beside your ears.", needs_view="side")
    return Finding("upper arm", "good", "Upper arms stay vertical and still.", needs_view="side")


def elbow_flare(m: Metrics, g: Geometry) -> Finding:
    w = m["elbow_width_x_shoulders"]
    if w <= 1.4:
        return Finding("elbow flare", "good", "Elbows stay tucked in.", needs_view="front")
    return Finding("elbow flare", "warn" if w <= 1.8 else "fault",
                   f"Elbows flare out to {w:.1f}x shoulder width. Pull them in so they point "
                   "forward, not out to the sides.", needs_view="front")


def lean_back(m: Metrics, g: Geometry) -> Finding:
    a = m["torso_lean_max"]
    if a < 10:
        return Finding("back arch", "good", "Torso stays upright.", needs_view="side")
    return Finding("back arch", "warn" if a < 18 else "fault",
                   f"Leaning back {a:.0f} deg. Keep ribs down and glutes tight so the lower back "
                   "does not arch.", needs_view="side")


def tempo(m: Metrics, g: Geometry) -> Finding:
    return tempo_finding(m, "lowering", "extending", 0.6)


LIFT = Lift(
    name="triceps_extension",
    view="front",
    signals=signals,
    rep_signal="elbow",
    metrics=metrics,
    checks=[stretch, lockout, elbow_flare, upper_arm, lean_back, tempo],
    chart=[Series("hand", "Hand height above shoulder", "%"), Series("elbow", "Elbow", tag="elbow"),
           Series("torso_lean", "Torso lean")],
    phases=("lowering", "bottom", "extending"),
    joints=("shoulder", "elbow"),   # the wrists are hidden behind the head for half of every rep
    path="hand",
)
