"""Barbell deadlift, conventional or sumo, from the floor. Film from the side."""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Profile, Series, Signals

# Wrist and foot tracking below this is a guess (from the side the near plate covers the hands and
# feet while the bar is low), so the checks that use them stand down.
MIN_BAR_VIS = 0.5


def signals(t: PoseTrack, g: Geometry) -> Signals:
    n, im = g.near, t.image
    sh, hip, knee, wr = (im[:, G.JOINTS[j][n]] for j in ("shoulder", "hip", "knee", "wrist"))
    # Facing direction from the knees, which sit ahead of the hips whenever the lifter is bent over;
    # the feet usually used for this are often hidden behind the plate.
    bent = g.hip_angle < 140
    dx = (knee[:, 0] - hip[:, 0])[bent] if bent.any() else knee[:, 0] - hip[:, 0]
    facing = float(np.sign(np.median(dx)) or g.facing)
    femur_px = float(np.median(np.linalg.norm(hip - knee, axis=-1))) + 1e-9
    m_per_px = g.femur / femur_px
    vis = t.vis
    bar_vis = np.minimum(vis[:, G.JOINTS["wrist"][n]],
                         vis[:, [G.JOINTS["heel"][n], G.JOINTS["toe"][n]]].mean(axis=1))
    return {
        "hip": g.hip_angle,
        "knee": g.knee_angle,
        # Torso from vertical, signed: + = leaning forward, - = leaning back.
        "lean": G.tilt(t, "hip", "shoulder", n) * facing,
        "shoulder_ahead_cm": facing * (sh[:, 0] - wr[:, 0]) * m_per_px * 100,
        "bar_above_knee": (knee[:, 1] - wr[:, 1]) / femur_px,   # image y points down
        "bar_offset_cm": g.bar_offset * 100,
        "bar_vis": bar_vis,
        # Reps are counted as excursions from the floor position up to lockout and back.
        "rep": -g.hip_angle,
    }


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b = r.bottom
    # The rep is counted from where the hips start opening; the set-up position is just before.
    s = max(0, r.start - int(fps))
    s += int(np.argmin(sig["hip"][s:b + 1]))
    above = np.nonzero(sig["bar_above_knee"][s:b + 1] >= 0)[0]
    k = s + int(above[0]) if len(above) else (s + b) // 2   # bar passes the knees
    w = int(fps * 0.5)
    top = slice(max(r.start, b - w), min(r.end, b + w) + 1)
    pull = slice(s, b + 1)
    return {
        "hip_angle_start": float(sig["hip"][s]),
        "knee_angle_start": float(sig["knee"][s]),
        "torso_lean_start": float(sig["lean"][s]),
        "lean_increase_off_floor": float(sig["lean"][s:k + 1].max() - sig["lean"][s]),
        "shoulder_ahead_of_bar_cm": float(sig["shoulder_ahead_cm"][s]),
        "bar_offset_start_cm": float(sig["bar_offset_cm"][s]),
        "bar_drift_forward_cm": float(sig["bar_offset_cm"][pull].max() - sig["bar_offset_cm"][s]),
        "bar_visibility": float(np.median(sig["bar_vis"][pull])),
        "hip_angle_top": float(sig["hip"][top].max()),
        "knee_angle_top": float(sig["knee"][top].max()),
        "lean_back_top": float(-sig["lean"][top].min()),
        "pulling_s": (b - s) / fps,
        "lowering_s": (r.end - b) / fps,
    }


def hips_first(m: Metrics, g: Geometry, p: Profile) -> Finding:
    d = m["lean_increase_off_floor"]
    if d < p.tol(5):
        return Finding("hip/chest timing", "good", "Hips and shoulders rise together off the floor.")
    return Finding("hip/chest timing", "warn" if d < p.tol(10) else "fault",
                   f"Torso tips {d:.0f} deg further forward before the bar reaches the knees; the "
                   "hips are rising before the bar moves. Push the floor away with your legs and keep "
                   "the chest up until the bar passes the knees.")


def start_position(m: Metrics, g: Geometry, p: Profile) -> Finding:
    if m["bar_visibility"] < MIN_BAR_VIS:
        return Finding("start position", "info", "Not assessed: the hands are hidden at the start "
                       "(usually by the near plate).", needs_view="side")
    a, hip = m["shoulder_ahead_of_bar_cm"], m["hip_angle_start"]
    # Sumo: the wide stance lets the torso sit more upright, shoulders over or just behind the bar.
    if a < -p.tol(6 if p.variation == "sumo" else 3):
        return Finding("start position", "warn", f"Shoulders start {-a:.0f} cm behind the bar (hip "
                       f"angle {hip:.0f} deg): hips too low, squatting the weight up. Raise the hips "
                       "until the shoulders are over or just ahead of the bar.", needs_view="side")
    return Finding("start position", "good", f"Shoulders start {a:.0f} cm ahead of the bar (hip "
                   f"angle {hip:.0f} deg).", needs_view="side")


def bar_path(m: Metrics, g: Geometry, p: Profile) -> Finding:
    if m["bar_visibility"] < MIN_BAR_VIS:
        return Finding("bar path", "info", "Not assessed: the hands or feet are hidden while the bar "
                       "is low (usually by the near plate). Film from slightly in front of side-on.",
                       needs_view="side")
    off, drift = m["bar_offset_start_cm"], m["bar_drift_forward_cm"]
    half_foot = p.tol(g.foot * 100 / 2)
    if abs(off) > half_foot:
        where = "ahead of" if off > 0 else "behind"
        return Finding("bar path", "warn", f"Bar starts {abs(off):.0f} cm {where} midfoot. Set up "
                       "with the bar over the middle of the foot, about an inch from the shins.",
                       needs_view="side")
    if drift > p.tol(5):
        return Finding("bar path", "warn", f"Bar drifts {drift:.0f} cm forward on the way up. Keep "
                       "the lats tight and drag it up the legs.", needs_view="side")
    return Finding("bar path", "good", f"Bar starts over midfoot and stays close (drifts "
                   f"{max(drift, 0):.0f} cm).", needs_view="side")


def lockout(m: Metrics, g: Geometry, p: Profile) -> Finding:
    hip, knee = m["hip_angle_top"], m["knee_angle_top"]
    if hip >= p.limit(180, 165) and knee >= p.limit(180, 160):
        return Finding("lockout", "good", f"Full lockout (hips {hip:.0f} deg, knees {knee:.0f} deg).")
    return Finding("lockout", "warn", f"Stops short of lockout (hips {hip:.0f} deg, knees {knee:.0f} "
                   "deg). Finish standing tall with the hips all the way through.")


def lean_back(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["lean_back_top"]
    if a < p.tol(8):
        return Finding("lean back", "good", "Stands straight at the top.", needs_view="side")
    return Finding("lean back", "warn" if a < p.tol(15) else "fault",
                   f"Leaning back {a:.0f} deg at the top, which loads the lower back for nothing. "
                   "Stand tall with glutes squeezed; the lift is done when hips and knees are straight.",
                   needs_view="side")


def tempo(m: Metrics, g: Geometry, p: Profile) -> Finding:
    return Finding("tempo", "info", f"Pulling {m['pulling_s']:.1f}s, lowering {m['lowering_s']:.1f}s.")


LIFT = Lift(
    name="deadlift",
    view="side",
    signals=signals,
    rep_signal="rep",
    metrics=metrics,
    checks=[hips_first, start_position, bar_path, lockout, lean_back, tempo],
    chart=[Series("hip", "Hip", tag="hip"), Series("knee", "Knee", tag="knee"),
           Series("lean", "Torso lean")],
    phases=("pulling", "lockout", "lowering"),
    min_dip=40.0,
    joints=("shoulder", "hip", "knee"),
    planted_feet=False,   # from the side the plates hide the feet
    variations={"conventional": "feet about hip width, hands outside the knees (default)",
                "sumo": "wide stance, hands inside the knees: a more upright start is expected"},
    camera="Put the camera at hip height, square to your side, far enough back that the whole bar "
           "and your feet stay in frame. Hands and feet hidden by the plate make the bar checks stand down.",
)
