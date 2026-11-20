"""Bent-over row with dumbbells or a bar. Film from the side for hinge, torso and pull height; from
the front for left/right symmetry."""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Profile, Series, Signals, tempo_finding

# Torso lean (from vertical) that counts as hinged for each style. A Pendlay row is done with the
# back about parallel to the floor and the bar returning to the floor every rep.
HINGE_BY_VARIATION = {"standard": (40.0, 25.0), "pendlay": (70.0, 55.0)}   # good from, fault below


def signals(t: PoseTrack, g: Geometry) -> Signals:
    n, im = g.near, t.image
    elbow = G.elbow_angle(t, g, n, "sagittal")
    # From the front the elbow bends toward the camera, which the picture barely shows; the hands
    # rising toward the torso show plainly from any angle. Rep signal: hand height, flipped so that
    # hands hanging is the resting level (segmentation looks for dips).
    hand = np.mean([G.hand_height(t, s) for s in (0, 1)], axis=0)
    upper = [np.linalg.norm(im[:, G.JOINTS["shoulder"][s]] - im[:, G.JOINTS["elbow"][s]], axis=-1) for s in (0, 1)]
    ua = np.percentile(np.r_[upper[0], upper[1]], 90) + 1e-9
    # Front view: left elbow above the right (+) as % of upper-arm length; forearm from vertical.
    elbow_diff = 100 * (im[:, G.JOINTS["elbow"][1], 1] - im[:, G.JOINTS["elbow"][0], 1]) / ua
    forearm = [np.abs(G.tilt(t, "elbow", "wrist", s)) for s in (0, 1)]
    # tilt is from straight up; a hanging forearm points down, so fold to 0 = vertical.
    forearm = np.maximum(*[np.minimum(f, 180 - f) for f in forearm])
    return {"elbow": elbow, "torso_lean": g.torso_lean, "upper_arm": G.arm_angle(t, g, n, "sagittal"),
            "hand": hand, "elbow_diff": elbow_diff, "forearm": forearm,
            "rep": elbow if g.view == "side" else -hand}


def posture(sig: Signals, g: Geometry) -> np.ndarray:
    """Hinged over = working position. Standing up between sets is not a row done too upright."""
    return sig["torso_lean"]


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b, whole = r.bottom, slice(r.start, r.end + 1)
    return {
        "elbow_angle_top": float(sig["elbow"][b]),
        "elbow_angle_bottom": float(max(sig["elbow"][r.start], sig["elbow"][r.end])),
        "upper_arm_past_torso": float(-sig["upper_arm"][b]),
        "torso_lean": float(np.median(sig["torso_lean"][whole])),
        "torso_movement": float(np.ptp(sig["torso_lean"][whole])),
        "elbow_height_difference_pct": float(abs(sig["elbow_diff"][b])),
        "forearm_from_vertical_top": float(sig["forearm"][b]),
        "pulling_s": (b - r.start) / fps,
        "lowering_s": (r.end - b) / fps,
    }


def hinge(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["torso_lean"]
    good, fault = HINGE_BY_VARIATION.get(p.variation, HINGE_BY_VARIATION["standard"])
    good, fault = p.limit(90, good), p.limit(90, fault)
    if a >= good:
        return Finding("hinge", "good", f"Torso hinged {a:.0f} deg forward.", needs_view="side")
    if p.variation == "pendlay":
        return Finding("hinge", "warn" if a >= fault else "fault",
                       f"Torso {a:.0f} deg forward; a Pendlay row is pulled with the back about "
                       "parallel to the floor. Hinge further and reset the bar on the floor each rep.",
                       needs_view="side")
    return Finding("hinge", "warn" if a >= fault else "fault",
                   f"Too upright (torso only {a:.0f} deg forward), which shifts the work to the "
                   "traps. Push your hips back until the chest is 45 deg or more toward the floor.",
                   needs_view="side")


def torso_still(m: Metrics, g: Geometry, p: Profile) -> Finding:
    # The shoulder itself travels as the shoulder blade pulls back, which reads as 10-15 deg of
    # torso movement on a strict row, so only more than that counts.
    d = m["torso_movement"]
    # Torso angle from the front is the 3D estimate's guess: it flagged 84% of the Mendeley
    # "good" front-view rows. Side view only.
    if d < p.tol(18):
        return Finding("torso movement", "good", "Torso angle holds steady.", needs_view="side")
    return Finding("torso movement", "warn" if d < p.tol(30) else "fault",
                   f"Torso angle changes {d:.0f} deg during the rep; you are heaving the weight "
                   "with your back. Hold the hinge still and pull with the arms and lats.",
                   needs_view="side")


def pull_height(m: Metrics, g: Geometry, p: Profile) -> Finding:
    past, el = m["upper_arm_past_torso"], m["elbow_angle_top"]
    if past >= -p.tol(5):
        return Finding("pull height", "good", "Elbows reach the torso at the top.", needs_view="side")
    return Finding("pull height", "warn" if past >= -p.tol(20) else "fault",
                   f"Short pull: elbow stops {-past:.0f} deg in front of the torso (elbow bent to "
                   f"{el:.0f} deg). Drive the elbows back past your ribs.", needs_view="side")


def full_extension(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["elbow_angle_bottom"]
    want = 160 if p.variation == "pendlay" else 150
    if a >= p.limit(180, want):
        return Finding("bottom range", "good", f"Arms straighten at the bottom ({a:.0f} deg).")
    return Finding("bottom range", "warn", f"Arms stay bent at the bottom ({a:.0f} deg). Let the "
                   "weight hang to a full stretch before each pull.")


def symmetry(m: Metrics, g: Geometry, p: Profile) -> Finding:
    d = m["elbow_height_difference_pct"]
    if d < p.tol(15):
        return Finding("pull symmetry", "good", "Both elbows reach the same height.", needs_view="front")
    return Finding("pull symmetry", "warn" if d < p.tol(30) else "fault",
                   f"One elbow finishes higher than the other (by {d:.0f}% of upper-arm length). "
                   "Pull both sides evenly; check your grip is centred.", needs_view="front")


def forearms(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["forearm_from_vertical_top"]
    if a < p.tol(20):
        return Finding("forearm line", "good", "Forearms stay vertical under the elbows.", needs_view="front")
    return Finding("forearm line", "warn", f"Forearms angle {a:.0f} deg off vertical at the top; the "
                   "elbows flare out or the hands drift wide. Keep the wrists under the elbows.",
                   needs_view="front")


def tempo(m: Metrics, g: Geometry, p: Profile) -> Finding:
    return tempo_finding(m, "pulling", "lowering", 0.5)


LIFT = Lift(
    name="bent_over_row",
    view="side",
    signals=signals,
    rep_signal="rep",
    metrics=metrics,
    checks=[hinge, torso_still, pull_height, full_extension, symmetry, forearms, tempo],
    chart=[Series("elbow", "Elbow", tag="elbow"), Series("torso_lean", "Torso lean", tag="hip"),
           Series("upper_arm", "Upper arm from torso", tag="shoulder")],
    also_views=("front",),
    front_chart=[Series("hand", "Hand height above shoulder", "%"),
                 Series("elbow_diff", "Left elbow above right", "%", tag="elbow"),
                 Series("forearm", "Forearm from vertical", tag="wrist")],
    phases=("pulling", "top", "lowering"),
    min_dip=30.0,
    joints=("shoulder", "elbow", "wrist", "hip"),
    path="hand",
    planted_feet=False,   # from the side the plates hide the feet
    variations={"standard": "torso hinged 45 deg or more (default)",
                "pendlay": "back parallel to the floor, bar back on the floor every rep"},
    posture=posture,
)
