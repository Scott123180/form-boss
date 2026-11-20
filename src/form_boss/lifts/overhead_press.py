"""Standing overhead (shoulder) press with dumbbells or a bar. Film from the front."""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Profile, Series, Signals, tempo_finding


def signals(t: PoseTrack, g: Geometry) -> Signals:
    sides = (g.near,) if g.view == "side" else (0, 1)   # from the side the far arm is hidden
    h = [G.hand_height(t, s) for s in (0, 1)]
    hand = np.mean([h[s] for s in sides], axis=0)
    elbow = np.mean([G.elbow_angle(t, g, s, "frontal") for s in sides], axis=0)
    # Torso from vertical in the image. From the side + = leaning back; otherwise it is the
    # sideways lean, whichever way.
    lean = (-G.tilt(t, "hip", "shoulder", g.near) * g.facing if g.view == "side"
            else np.abs(G.tilt_mid(t, "hip", "shoulder")))
    # Reps are counted from the hands rising above the shoulders, so standing with the arms
    # down between sets reads the same as holding the weights at the shoulders.
    return {"hand_l": h[0], "hand_r": h[1], "hand": hand, "elbow": elbow, "torso_lean": lean,
            "rep": -np.maximum(hand, 0)}


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b = r.bottom
    return {
        "hand_height_top_pct": float(sig["hand"][b]),
        "hand_height_bottom_pct": float(min(sig["hand"][r.start], sig["hand"][r.end])),
        "elbow_angle_top": float(sig["elbow"][b]),
        "left_right_difference_pct": float(abs(sig["hand_l"][b] - sig["hand_r"][b])),
        "torso_lean_max": float(sig["torso_lean"][r.start:r.end + 1].max()),
        "pressing_s": (b - r.start) / fps,
        "lowering_s": (r.end - b) / fps,
    }


def lockout(m: Metrics, g: Geometry, p: Profile) -> Finding:
    h, a = m["hand_height_top_pct"], m["elbow_angle_top"]
    if h >= p.limit(100, 85) and a >= p.limit(180, 150):
        return Finding("lockout", "good", f"Arms lock out overhead (elbows {a:.0f} deg).")
    if h >= p.limit(100, 70):
        return Finding("lockout", "warn", f"Just short of lockout (elbows {a:.0f} deg, hands at "
                       f"{h:.0f}% of full reach). Finish with straight arms and biceps by your ears.")
    return Finding("lockout", "fault", f"Press stops well short (hands reach {h:.0f}% of full "
                   "overhead height). Press until the arms are straight; go lighter if needed.")


def bottom_range(m: Metrics, g: Geometry, p: Profile) -> Finding:
    h = m["hand_height_bottom_pct"]
    # With the upper arms level with the shoulders the hands sit about half an arm length above them.
    if h <= p.limit(30, 50):
        return Finding("bottom range", "good", "Elbows come down to shoulder level or below between reps.")
    return Finding("bottom range", "warn" if h <= p.limit(30, 65) else "fault",
                   f"Lowering stops high (hands still {h:.0f}% of arm length above the shoulders). "
                   "Bring the weights down until they are level with your ears or lower.")


def symmetry(m: Metrics, g: Geometry, p: Profile) -> Finding:
    d = m["left_right_difference_pct"]
    if d < p.tol(10):
        return Finding("symmetry", "good", "Both hands finish at the same height.", needs_view="front")
    return Finding("symmetry", "warn", f"One hand finishes higher than the other (difference {d:.0f}% "
                   "of arm length). Press both sides evenly.", needs_view="front")


def lean_back(m: Metrics, g: Geometry, p: Profile) -> Finding:
    a = m["torso_lean_max"]
    if a < p.tol(10):
        return Finding("back lean", "good", "Torso stays upright.", needs_view="side")
    return Finding("back lean", "warn" if a < p.tol(18) else "fault",
                   f"Leaning back {a:.0f} deg to get the weight up, which loads the lower back. "
                   "Squeeze glutes, keep ribs down and press straight up.", needs_view="side")


def tempo(m: Metrics, g: Geometry, p: Profile) -> Finding:
    return tempo_finding(m, "pressing", "lowering", 0.6)


LIFT = Lift(
    name="overhead_press",
    view="front",
    signals=signals,
    rep_signal="rep",
    metrics=metrics,
    checks=[lockout, bottom_range, symmetry, lean_back, tempo],
    chart=[Series("hand", "Hand height above shoulder", "%"), Series("elbow", "Elbow", tag="elbow"),
           Series("torso_lean", "Torso lean")],
    phases=("pressing", "lockout", "lowering"),
    min_dip=35.0,
    near=30.0,
    joints=("shoulder", "elbow", "wrist"),
    path="hand",
)
