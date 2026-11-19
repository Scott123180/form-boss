"""Barbell back squat."""

from __future__ import annotations

import numpy as np

from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Series, Signals


def signals(t: PoseTrack, g: Geometry) -> Signals:
    return {"knee": g.knee_angle, "torso_lean": g.torso_lean, "thigh": g.thigh_angle}


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> dict[str, float]:
    b = r.bottom
    asc = slice(b, max(b + 1, r.end))
    # "Hips shooting up": how much further the torso tips forward after leaving the bottom.
    lean_rise = float(g.torso_lean[asc].max() - g.torso_lean[b])
    whole = slice(r.start, r.end + 1)
    return {
        "knee_angle_bottom": float(g.knee_angle[b]),
        "hip_angle_bottom": float(g.hip_angle[b]),
        "thigh_angle_bottom": float(g.thigh_angle[b]),
        "hip_below_knee_cm": float(g.hip_below_knee[b] * 100),
        "torso_lean_bottom": float(g.torso_lean[b]),
        "torso_lean_max": float(g.torso_lean[whole].max()),
        "lean_increase_on_ascent": lean_rise,
        "bar_offset_bottom_cm": float(g.bar_offset[b] * 100),
        "bar_travel_fwd_back_cm": float(np.ptp(g.bar_offset[whole]) * 100),
        "descent_s": (b - r.start) / fps,
        "ascent_s": (r.end - b) / fps,
    }


# Thigh angle is measured between joint centres. The usual standard, hip crease below the top of
# the knee, is met a little before the joint centres are level: the top of the knee sits roughly
# 4-5 cm above the knee joint, about 6 degrees of thigh angle for a typical femur.
DEPTH_OK_DEG = 5.0
DEPTH_BORDERLINE_DEG = 12.0


def depth(m: dict[str, float], g: Geometry) -> Finding:
    a = m["thigh_angle_bottom"]
    desc = f"thigh {abs(a):.0f} deg {'above' if a > 0 else 'below'} horizontal"
    if a <= DEPTH_OK_DEG:
        return Finding("depth", "good", f"Reached depth ({desc}).")
    if a <= DEPTH_BORDERLINE_DEG:
        return Finding("depth", "warn", f"Borderline depth ({desc}). Sink slightly deeper so the "
                       "hip crease clearly drops below the top of the knee.")
    return Finding("depth", "fault", f"High ({desc}). Squat deeper until the hip crease is below "
                   "the top of the knee.")


def hips_shoot_up(m: dict[str, float], g: Geometry) -> Finding:
    d = m["lean_increase_on_ascent"]
    if d < 5:
        return Finding("hip/chest timing", "good", "Hips and chest rise together.")
    status = "warn" if d < 10 else "fault"
    return Finding("hip/chest timing", status,
                   f"Torso tips {d:.0f} deg further forward out of the bottom; hips are rising "
                   "faster than the chest. Drive the upper back into the bar as you stand.")


def torso_lean(m: dict[str, float], g: Geometry) -> Finding:
    lean = m["torso_lean_max"]
    # Longer femurs relative to the torso need more forward lean to keep the bar over midfoot.
    expected = 35 + 25 * (g.femur / g.torso - 0.8)
    if lean > expected + 20:
        return Finding("torso lean", "warn",
                       f"Max forward lean {lean:.0f} deg, well beyond the ~{expected:.0f} deg "
                       "expected for your proportions. Check the bar is not drifting forward.")
    return Finding("torso lean", "info",
                   f"Max forward lean {lean:.0f} deg (about {expected:.0f} deg expected for your "
                   "femur/torso ratio).")


def bar_over_midfoot(m: dict[str, float], g: Geometry) -> Finding:
    off, travel = m["bar_offset_bottom_cm"], m["bar_travel_fwd_back_cm"]
    half_foot = g.foot * 100 / 2
    where = "ahead of" if off > 0 else "behind"
    msg = f"Bar {abs(off):.0f} cm {where} midfoot at the bottom; {travel:.0f} cm fore/aft travel."
    if abs(off) > half_foot:
        cue = ("Keep weight on the whole foot and sit back." if off > 0
               else "You are on your heels; keep the bar over the middle of the foot.")
        return Finding("bar over midfoot", "warn", f"{msg} {cue}", needs_view="side")
    return Finding("bar over midfoot", "good", msg, needs_view="side")


def tempo(m: dict[str, float], g: Geometry) -> Finding:
    d, a = m["descent_s"], m["ascent_s"]
    if d < 0.8:
        return Finding("tempo", "warn", f"Descent {d:.1f}s, ascent {a:.1f}s. That is a fast "
                       "drop; control the way down.")
    return Finding("tempo", "info", f"Descent {d:.1f}s, ascent {a:.1f}s.")


LIFT = Lift(
    name="squat",
    view="side",
    signals=signals,
    rep_signal="knee",
    metrics=metrics,
    checks=[depth, hips_shoot_up, torso_lean, bar_over_midfoot, tempo],
    chart=[Series("knee", "Knee", tag="knee"), Series("torso_lean", "Torso lean", tag="hip"),
           Series("thigh", "Thigh above horizontal")],
    # Unracking, racking and stepping under the bar bend the knees as far as a rep but take ~0.2 s each
    # way. The quickest real reps in the Mendeley clips take ~0.3 s each way, timed the same way.
    min_rep_s=0.6,
    min_phase_s=0.25,
    # Ducking under the bar and walking it out are slow but bend the knees ~40-45 deg against ~90 for
    # a working rep on the user clips. Dips this shallow in the Mendeley clips are tracking errors.
    min_depth_frac=0.6,
)
