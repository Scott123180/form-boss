"""Barbell back squat. Film from the side for depth, torso and bar path; from the front or back for
knee tracking, hip shift and bar level."""

from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep
from .base import Finding, Lift, Profile, Series, Signals

# Expected max forward lean for a typical femur/torso ratio, by bar position: a low bar sits further
# down the back and needs more lean to stay over midfoot; a front squat needs an upright torso.
LEAN_BY_VARIATION = {"high_bar": 35.0, "low_bar": 45.0, "front": 25.0}


def signals(t: PoseTrack, g: Geometry) -> Signals:
    im = t.image
    # Hip height above the ankles as % of standing height: a rep signal any camera angle can see.
    # Measured against a body length the squat does not change in that view (shoulder width from
    # the front, torso length otherwise), and against the standing height over the surrounding
    # seconds, because the lifter walks toward or away from the camera between reps.
    leg = np.linalg.norm(G.mid(t, "hip") - G.mid(t, "ankle"), axis=-1)
    scale = (np.abs(im[:, G.JOINTS["shoulder"][0], 0] - im[:, G.JOINTS["shoulder"][1], 0]) if g.view == "front"
             else np.linalg.norm(G.mid(t, "shoulder") - G.mid(t, "hip"), axis=-1))
    hip_height = 100 * G.relative_to_standing(leg / (scale + 1e-9), t.fps)
    width = lambda j: np.abs(im[:, G.JOINTS[j][0], 0] - im[:, G.JOINTS[j][1], 0])
    # Floor the stance width: turning side-on to walk to the rack brings the ankles together in the
    # picture, and dividing by that blows up.
    floor = lambda x: np.maximum(x, 0.5 * np.median(x[t.found]))
    ankle_w = floor(width("ankle"))
    # Front/rear view. Knee width over ankle width: knees over the feet read about 1; below 1 the
    # knees are caving in. Hip centre left/right of the feet's centre, % of stance width.
    knee_track = (G.width_at_hips(t, "knee") / floor(G.width_at_hips(t, "ankle"))
                  if g.view == "front" else width("knee") / ankle_w)
    hip_shift = 100 * (G.mid(t, "hip")[:, 0] - G.mid(t, "ankle")[:, 0]) / ankle_w
    # Bar tilt: wrist line against the line of the feet, which are on level ground, so a camera
    # held crooked does not read as a tilted bar. Signed degrees.
    def line(j):
        d = im[:, G.JOINTS[j][1]] - im[:, G.JOINTS[j][0]]
        d = d * np.sign(d[:, :1] + 1e-9)   # point both lines the same way across the picture
        return np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    bar_tilt = line("wrist") - np.median(line("ankle"))
    # Hands on a bar across the back sit wider than the shoulders; a bodyweight squat's do not.
    grip = width("wrist") / floor(width("shoulder"))
    # Rep signal. From the side, the knee angle itself. Otherwise hip height on the knee-angle
    # scale, so the same rep thresholds apply: a parallel squat drops the hips to about half their
    # standing height and bends the knee about 90 deg, so 2 deg per %.
    knee = g.knee_angle if g.view == "side" else 180 - 2 * (100 - hip_height)
    # Hand height of the better-seen wrist (from the side the far one is hidden and wanders).
    seen_r = t.vis[:, G.JOINTS["wrist"][1]] > t.vis[:, G.JOINTS["wrist"][0]]
    hands = np.where(seen_r, G.hand_height(t, 1), G.hand_height(t, 0))
    return {"knee": g.knee_angle, "torso_lean": g.torso_lean, "thigh": g.thigh_angle, "hands": hands,
            "hip_height": hip_height, "knee_track": knee_track, "knee_track_pct": 100 * knee_track,
            "hip_shift": hip_shift,
            "bar_tilt": bar_tilt, "grip": grip, "rep": knee}


def posture(sig: Signals, g: Geometry) -> np.ndarray:
    """Hands up at the shoulders, holding the bar: 0 = arms hanging, ~100 = at shoulder height.
    A knee bend with the arms down in a barbell clip is the lifter without the bar (fixing a belt,
    picking something up). Relative to the clip's reps, so bodyweight squats are unaffected."""
    return np.clip(sig["hands"] + 100, 0, 100)


def metrics(sig: Signals, g: Geometry, r: Rep, fps: float) -> dict[str, float]:
    b = r.bottom
    asc = slice(b, max(b + 1, r.end))
    # "Hips shooting up": how much further the torso tips forward after leaving the bottom.
    lean_rise = float(g.torso_lean[asc].max() - g.torso_lean[b])
    whole = slice(r.start, r.end + 1)
    # Front view measurements: the bottom and the way up, where knees cave and hips shift.
    low = slice(b, max(b + 1, b + (r.end - b) // 2 + 1))
    shift0 = float(np.median(sig["hip_shift"][max(0, r.start - int(fps * 0.3)):r.start + 1]))
    return {
        "knee_angle_bottom": float(g.knee_angle[b]),
        "hip_angle_bottom": float(g.hip_angle[b]),
        "thigh_angle_bottom": float(g.thigh_angle[b]),
        "hip_below_knee_cm": float(g.hip_below_knee[b] * 100),
        "hip_height_bottom_pct": float(sig["hip_height"][b]),
        "torso_lean_bottom": float(g.torso_lean[b]),
        "torso_lean_max": float(g.torso_lean[whole].max()),
        "lean_increase_on_ascent": lean_rise,
        "bar_offset_bottom_cm": float(g.bar_offset[b] * 100),
        "bar_travel_fwd_back_cm": float(np.ptp(g.bar_offset[whole]) * 100),
        "knee_track_bottom": float(sig["knee_track"][b]),
        "knee_track_min": float(sig["knee_track"][low].min()),
        "hip_shift_max_pct": float(np.max(np.abs(sig["hip_shift"][low] - shift0))),
        "bar_tilt_max": float(np.max(np.abs(sig["bar_tilt"][whole]))),
        "grip_width_x_shoulders": float(np.median(sig["grip"][whole])),
        "hand_height_pct": float(np.median(sig["hands"][whole])),
        "descent_s": (b - r.start) / fps,
        "ascent_s": (r.end - b) / fps,
    }


# Thigh angle is measured between joint centres. The usual standard, hip crease below the top of
# the knee, is met a little before the joint centres are level: the top of the knee sits roughly
# 4-5 cm above the knee joint, about 6 degrees of thigh angle for a typical femur.
DEPTH_OK_DEG = 5.0
DEPTH_BORDERLINE_DEG = 12.0


def depth(m: dict[str, float], g: Geometry, p: Profile) -> Finding:
    a = m["thigh_angle_bottom"]
    desc = f"thigh {abs(a):.0f} deg {'above' if a > 0 else 'below'} horizontal"
    if g.view != "side":
        # Thigh angle cannot be read from the front; hip height is a rough stand-in, and too
        # dependent on camera height to grade (the same squats read 50% from behind at knee height
        # and 68% from in front at the floor), so it is only reported.
        h = m["hip_height_bottom_pct"]
        return Finding("depth", "info", f"Hips drop to {h:.0f}% of their standing height (about "
                       "50% is parallel). Film from the side to judge depth.", needs_view="side")
    if a <= p.tol(DEPTH_OK_DEG):
        return Finding("depth", "good", f"Reached depth ({desc}).")
    if a <= p.tol(DEPTH_BORDERLINE_DEG):
        return Finding("depth", "warn", f"Borderline depth ({desc}). Sink slightly deeper so the "
                       "hip crease clearly drops below the top of the knee.")
    return Finding("depth", "fault", f"High ({desc}). Squat deeper until the hip crease is below "
                   "the top of the knee.")


def hips_shoot_up(m: dict[str, float], g: Geometry, p: Profile) -> Finding:
    d = m["lean_increase_on_ascent"]
    if g.view == "front":
        # From the front the torso angle is the 3D estimate's guess; it flagged good and bad
        # Mendeley front-view squats alike.
        return Finding("hip/chest timing", "info", "Not assessed: needs a side or angled view.")
    if d < p.tol(5):
        return Finding("hip/chest timing", "good", "Hips and chest rise together.")
    status = "warn" if d < p.tol(10) else "fault"
    return Finding("hip/chest timing", status,
                   f"Torso tips {d:.0f} deg further forward out of the bottom; hips are rising "
                   "faster than the chest. Drive the upper back into the bar as you stand.")


def torso_lean(m: dict[str, float], g: Geometry, p: Profile) -> Finding:
    lean = m["torso_lean_max"]
    # Longer femurs relative to the torso need more forward lean to keep the bar over midfoot.
    base = LEAN_BY_VARIATION.get(p.variation, 35.0)
    expected = base + 25 * (g.femur / g.torso - 0.8)
    style = f" for a {p.variation.replace('_', '-')} squat" if p.variation != "high_bar" else ""
    if lean > expected + p.tol(20):
        return Finding("torso lean", "warn",
                       f"Max forward lean {lean:.0f} deg, well beyond the ~{expected:.0f} deg "
                       f"expected for your proportions{style}. Check the bar is not drifting forward.")
    return Finding("torso lean", "info",
                   f"Max forward lean {lean:.0f} deg (about {expected:.0f} deg expected for your "
                   f"femur/torso ratio{style}).")


def bar_over_midfoot(m: dict[str, float], g: Geometry, p: Profile) -> Finding:
    off, travel = m["bar_offset_bottom_cm"], m["bar_travel_fwd_back_cm"]
    half_foot = g.foot * 100 / 2
    where = "ahead of" if off > 0 else "behind"
    msg = f"Bar {abs(off):.0f} cm {where} midfoot at the bottom; {travel:.0f} cm fore/aft travel."
    if abs(off) > p.tol(half_foot):
        cue = ("Keep weight on the whole foot and sit back." if off > 0
               else "You are on your heels; keep the bar over the middle of the foot.")
        return Finding("bar over midfoot", "warn", f"{msg} {cue}", needs_view="side")
    return Finding("bar over midfoot", "good", msg, needs_view="side")


def knee_tracking(m: dict[str, float], g: Geometry, p: Profile) -> Finding:
    # Knees over the toes with the feet turned out put the knees wider than the ankles, ~1.0-1.4.
    # Perspective is corrected for, but the same knees-over-feet squats still read 0.85 from behind
    # and 0.97 from in front on the user clips, so only a clear cave inward is flagged.
    k, kb = m["knee_track_min"], m["knee_track_bottom"]
    if k >= p.limit(1.0, 0.8):
        return Finding("knee tracking", "good", f"Knees track out over the feet (knee width "
                       f"{k:.2f}x ankle width at the narrowest).", needs_view="front")
    cave = " as you drive up" if kb - k > 0.1 else ""
    return Finding("knee tracking", "warn" if k >= p.limit(1.0, 0.65) else "fault",
                   f"Knees cave inward{cave} (knee width drops to {k:.2f}x ankle width). Push the "
                   "knees out in line with the toes all the way up.", needs_view="front")


def hip_shift(m: dict[str, float], g: Geometry, p: Profile) -> Finding:
    s = m["hip_shift_max_pct"]
    if s < p.tol(10):
        return Finding("hip shift", "good", "Hips stay centred over the feet.", needs_view="front")
    return Finding("hip shift", "warn" if s < p.tol(20) else "fault",
                   f"Hips shift {s:.0f}% of your stance width to one side out of the bottom. Often a "
                   "mobility or strength imbalance; keep weight even on both feet.", needs_view="front")


def bar_level(m: dict[str, float], g: Geometry, p: Profile) -> Finding:
    # Hands on the bar are ~70-90 cm apart: 6 deg of tilt is 7-9 cm higher on one side.
    a = m["bar_tilt_max"]
    if m["grip_width_x_shoulders"] < 1.2 or abs(m["hand_height_pct"]) > 50:
        return Finding("bar level", "info", "Not assessed: the hands are not on a bar across the "
                       "shoulders.", needs_view="front")
    if a < p.tol(6):
        return Finding("bar level", "good", "Bar stays level across the shoulders.", needs_view="front")
    return Finding("bar level", "warn", f"Bar tilts up to {a:.0f} deg; one side drops or lags. "
                   "Centre the bar on your back and keep both hands set evenly.", needs_view="front")


def tempo(m: dict[str, float], g: Geometry, p: Profile) -> Finding:
    d, a = m["descent_s"], m["ascent_s"]
    if d < 0.8:
        return Finding("tempo", "warn", f"Descent {d:.1f}s, ascent {a:.1f}s. That is a fast "
                       "drop; control the way down.")
    return Finding("tempo", "info", f"Descent {d:.1f}s, ascent {a:.1f}s.")


LIFT = Lift(
    name="squat",
    view="side",
    signals=signals,
    rep_signal="rep",
    metrics=metrics,
    checks=[depth, hips_shoot_up, torso_lean, bar_over_midfoot, knee_tracking, hip_shift, bar_level,
            tempo],
    chart=[Series("knee", "Knee", tag="knee"), Series("torso_lean", "Torso lean", tag="hip"),
           Series("thigh", "Thigh above horizontal")],
    also_views=("front",),
    front_chart=[Series("hip_height", "Hip height", "%", tag="hip"),
                 Series("knee_track_pct", "Knee width, % of ankle width", "%", tag="knee"),
                 Series("hip_shift", "Hip shift", "%")],
    # Unracking, racking and stepping under the bar bend the knees as far as a rep but take ~0.2 s each
    # way. The quickest real reps in the Mendeley clips take ~0.3 s each way, timed the same way.
    min_rep_s=0.6,
    min_phase_s=0.25,
    # Ducking under the bar and walking it out are slow but bend the knees ~40-45 deg against ~90 for
    # a working rep on the user clips. Dips this shallow in the Mendeley clips are tracking errors.
    min_depth_frac=0.6,
    posture=posture,
    variations={"high_bar": "bar on top of the shoulders (default)",
                "low_bar": "bar lower on the rear delts: more forward lean is expected",
                "front": "bar in front on the shoulders: a more upright torso is expected"},
)
