"""Barbell bench press. Film from the side at bench height, a step toward the feet so the near plate
does not cover your shoulder. A camera at the head end (counted as a front view) checks bar tilt."""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Metrics, Series, Signals, tempo_finding


def signals(t: PoseTrack, g: Geometry) -> Signals:
    n = g.near
    sides = (n,) if g.view == "side" else (0, 1)   # from the side the far arm is hidden
    # Lying on the bench, "up" in the image is still up, so hand height above the shoulder reads
    # ~100% at lockout and drops to roughly chest thickness when the bar touches.
    h = [G.hand_height(t, s) for s in (0, 1)]
    hand = np.mean([h[s] for s in sides], axis=0)
    # A flared elbow bends in neither the side nor the front plane, so only lockout (straight or
    # not) is judged from it, and that reads fine straight off the image from any angle. The 3D
    # estimate was 30-40 deg short at lockout on lying lifters.
    elbow = np.mean([G.image_angle(t, "shoulder", "elbow", "wrist", s) for s in sides], axis=0)

    # From the side: where along the body the bar sits, as % of shoulder-to-hip length.
    # 0 = over the shoulder joint, + = toward the hips (lower chest), - = toward the face.
    im = t.image
    sh, hip, wr = (im[:, G.JOINTS[j][n]] for j in ("shoulder", "hip", "wrist"))
    axis = np.median(hip - sh, axis=0)
    torso_px = float(np.linalg.norm(axis)) + 1e-9
    along = 100 * ((wr - sh) @ (axis / torso_px)) / torso_px
    # Hip height above its usual level on the bench, % of torso length (image y points down).
    hip_rise = 100 * (np.median(hip[:, 1]) - hip[:, 1]) / torso_px
    # Bar tilt: angle of the wrist-to-wrist line in the image. Only meaningful from the head or
    # foot end, where the bar runs across the picture.
    d = im[:, G.JOINTS["wrist"][1]] - im[:, G.JOINTS["wrist"][0]]
    tilt = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0]) + 1e-9))
    return {"hand": hand, "elbow": elbow, "bar_along": along, "hip_rise": hip_rise, "bar_tilt": tilt}


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> Metrics:
    b, whole = r.bottom, slice(r.start, r.end + 1)
    # The rep starts and ends where the bar has come a little way off lockout, so look for the
    # lockout itself just outside those points.
    w = int(fps * 0.7)
    tops = np.r_[max(0, r.start - w):r.start + 1, r.end:min(len(sig["hand"]), r.end + w + 1)]
    top = int(tops[np.argmax(sig["hand"][tops])])
    return {
        "hand_height_bottom_pct": float(sig["hand"][b]),
        "hand_height_top_pct": float(sig["hand"][top]),
        "elbow_angle_top": float(sig["elbow"][tops].max()),
        "touch_point_pct": float(sig["bar_along"][b]),
        "finish_point_pct": float(sig["bar_along"][top]),
        "hip_rise_pct": float(sig["hip_rise"][whole].max() - sig["hip_rise"][r.start]),
        "bar_tilt_max": float(sig["bar_tilt"][whole].max()),
        "lowering_s": (b - r.start) / fps,
        "pressing_s": (r.end - b) / fps,
    }


def depth(m: Metrics, g: Geometry) -> Finding:
    # With the bar on the chest the wrists sit about a chest's depth above the shoulder joints,
    # roughly 20-30% of arm length.
    h = m["hand_height_bottom_pct"]
    if h <= 35:
        return Finding("depth", "good", "Bar comes down to the chest.")
    if h <= 50:
        return Finding("depth", "warn", f"Bar may stop short of the chest (hands still {h:.0f}% of "
                       "arm length above the shoulders). Touch the chest on every rep.")
    return Finding("depth", "fault", f"Half rep: bar stops well above the chest (hands {h:.0f}% of "
                   "arm length above the shoulders). Lower it until it touches; go lighter if needed.")


def lockout(m: Metrics, g: Geometry) -> Finding:
    a, h = m["elbow_angle_top"], m["hand_height_top_pct"]
    if a >= 155 and h >= 80:
        return Finding("lockout", "good", f"Arms lock out at the top (elbows {a:.0f} deg).")
    return Finding("lockout", "warn", f"Arms stay bent at the top (elbows {a:.0f} deg). Press to "
                   "straight arms before the next rep.")


def touch_point(m: Metrics, g: Geometry) -> Finding:
    p = m["touch_point_pct"]
    if p < 15:
        return Finding("touch point", "warn", f"Bar touches high on the chest ({p:.0f}% of the way "
                       "from shoulders to hips), which usually goes with elbows flared out to the "
                       "sides and is hard on the shoulders. Tuck the elbows somewhat and touch at "
                       "the lower chest.", needs_view="side")
    if p > 60:
        return Finding("touch point", "warn", f"Bar touches low, near the belly ({p:.0f}% of the way "
                       "from shoulders to hips). Bring it down to the bottom of the breastbone.",
                       needs_view="side")
    return Finding("touch point", "good", f"Bar touches the mid-to-lower chest ({p:.0f}% of the way "
                   "from shoulders to hips).", needs_view="side")


def finish_point(m: Metrics, g: Geometry) -> Finding:
    p = m["finish_point_pct"]
    if abs(p) <= 15:
        return Finding("bar path", "good", "Bar finishes over the shoulders.", needs_view="side")
    where = "toward your feet, over the chest" if p > 0 else "behind the shoulders, over the face"
    return Finding("bar path", "warn", f"Bar locks out {abs(p):.0f}% of torso length {where}. "
                   "Press up and slightly back so it finishes straight above the shoulders.",
                   needs_view="side")


def hips_down(m: Metrics, g: Geometry) -> Finding:
    d = m["hip_rise_pct"]
    if d < 8:
        return Finding("hips on bench", "good", "Hips stay on the bench.", needs_view="side")
    return Finding("hips on bench", "warn" if d < 15 else "fault",
                   f"Hips rise off the bench during the press (by {d:.0f}% of torso length). Keep "
                   "the glutes down and drive through the feet into the floor, not up.",
                   needs_view="side")


def bar_level(m: Metrics, g: Geometry) -> Finding:
    # Wrist tracking jitters by a few degrees on a level bar; 8 deg is ~10 cm end to end on a
    # typical grip, which is a visible tilt.
    a = m["bar_tilt_max"]
    if a < 8:
        return Finding("bar level", "good", "Bar stays level.", needs_view="front")
    return Finding("bar level", "warn", f"Bar tilts up to {a:.0f} deg; one side lags. Press both "
                   "sides evenly and check your grip is centred.", needs_view="front")


def tempo(m: Metrics, g: Geometry) -> Finding:
    return tempo_finding(m, "lowering", "pressing", 0.5)


LIFT = Lift(
    name="bench_press",
    view="side",
    signals=signals,
    rep_signal="hand",
    metrics=metrics,
    checks=[depth, lockout, touch_point, finish_point, hips_down, bar_level, tempo],
    chart=[Series("hand", "Hand height above shoulder", "%"), Series("elbow", "Elbow", tag="elbow"),
           Series("bar_along", "Bar toward hips", "%")],
    phases=("lowering", "chest", "pressing"),
    min_dip=35.0,
    joints=("shoulder", "elbow", "wrist", "hip"),
    camera="Put the camera at bench height, square to your side and a step toward your feet so the "
           "near plate does not cover your shoulder, with the whole bench in frame.",
)
