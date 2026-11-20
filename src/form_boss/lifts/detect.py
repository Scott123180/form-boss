"""Work out which lift a clip shows from how the body moves, for when --lift is left out.

The lift is decided by a few questions any camera angle can answer: is the lifter lying down or
on their feet, which joint does most of the moving (knees/hips or elbows, or the torso bending
sideways), and where the hands are (on the back, hanging, overhead, held above the chest).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import geometry as G
from ..geometry import Geometry
from ..pose import PoseTrack


@dataclass
class Guess:
    lift: str
    reason: str
    features: dict


def _range(x: np.ndarray) -> float:
    # Wide percentiles: in a long clip with a short set, the reps are a few percent of the frames.
    return float(np.percentile(x, 98) - np.percentile(x, 2)) if len(x) else 0.0


def features(t: PoseTrack, g: Geometry) -> dict[str, float]:
    f = t.found if t.found.sum() > 10 else np.ones(len(t), bool)
    im, w = t.image[f], t.world[f]
    J = G.JOINTS

    def mid(x, j):
        return (x[:, J[j][0]] + x[:, J[j][1]]) / 2

    # Thigh direction from vertical in 3D: ~75-90 lying on a bench or in a plank, under ~45 on
    # your feet even at the bottom of a squat. 3D so that a bench filmed from the foot end counts.
    th = mid(w, "knee") - mid(w, "hip")
    thigh = np.degrees(np.arccos(np.clip(th[:, 1] / (np.linalg.norm(th, axis=-1) + 1e-9), -1, 1)))
    # Hands toward the ceiling (+) or the floor (-) from the shoulders, 3D, in arm lengths.
    arm = float(np.median(np.linalg.norm(w[:, J["shoulder"][0]] - w[:, J["elbow"][0]], axis=-1)
                          + np.linalg.norm(w[:, J["elbow"][0]] - w[:, J["wrist"][0]], axis=-1))) + 1e-9
    hands_up = (mid(w, "shoulder")[:, 1] - mid(w, "wrist")[:, 1]) / arm   # world y points down
    # How far the hips drop toward the ankles in the picture, as a share of the standing height:
    # ~0.5 for a squat, ~0.3 for a deadlift, under 0.15 when the legs only hold position.
    leg = np.linalg.norm(mid(im, "hip") - mid(im, "ankle"), axis=-1)
    hip_drop = 1 - np.percentile(leg, 2) / (np.percentile(leg, 95) + 1e-9)
    hand = np.maximum(G.hand_height(t, 0), G.hand_height(t, 1))[f]
    # Elbow height above the shoulder in upper-arm lengths, the lower elbow: stays near +1 through
    # a triceps extension, drops below 0 at the bottom of every press.
    elbow_up = np.minimum(*[(im[:, J["shoulder"][s], 1] - im[:, J["elbow"][s], 1])
                            / (np.linalg.norm(im[:, J["shoulder"][s]] - im[:, J["elbow"][s]], axis=-1) + 1e-9)
                            for s in (0, 1)])
    # How far the elbows bend while the torso is hinged: rows bend them, deadlifts keep them straight.
    hinged = g.torso_lean[f] > 25
    # Measured in the picture and in 3D: from the front a row bends the elbow toward the camera.
    elbow_hinged = 0.0
    if hinged.sum() >= 10:
        wh = w[hinged]
        elbow_hinged = max(
            min(_range(G.image_angle(t, "shoulder", "elbow", "wrist", s)[f][hinged]),
                _range(G._angle(wh[:, J["shoulder"][s]], wh[:, J["elbow"][s]], wh[:, J["wrist"][s]])))
            if g.view == "side" else
            _range(G._angle(wh[:, J["shoulder"][s]], wh[:, J["elbow"][s]], wh[:, J["wrist"][s]]))
            for s in (0, 1))
    # Sideways torso tilt in the picture; only means sideways when filmed from the front or back.
    d = mid(im, "shoulder") - mid(im, "hip")
    side = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))
    return {
        "thigh_tilt_3d": float(np.median(thigh)),
        "hands_up_3d": float(np.median(hands_up)),
        "hip_drop": float(hip_drop),
        "hip_range": _range(g.hip_angle[f]),
        "hand_height_median": float(np.median(hand)),
        "elbow_above_shoulder_p10": float(np.percentile(elbow_up, 10)),
        "torso_lean_median": float(np.median(g.torso_lean[f])),
        "torso_lean_range": _range(g.torso_lean[f]),
        "elbow_range_hinged": float(elbow_hinged),
        "side_tilt_range": _range(side) if g.view != "side" else 0.0,
    }


def guess_lift(t: PoseTrack, g: Geometry) -> Guess:
    """Thresholds sit between the per-lift ranges measured on the Mendeley clips (all seven
    exercises, side and front views) and the user and sample barbell clips; see SOURCES.md."""
    x = features(t, g)

    def done(lift: str, why: str) -> Guess:
        return Guess(lift, why, x)

    if x["torso_lean_median"] > 60 and x["hands_up_3d"] > 0:
        return done("bench_press", "lying down with the hands above the chest")
    if x["thigh_tilt_3d"] > 50:   # standing lifts stay under ~45 even in a deep squat
        if x["hands_up_3d"] > 0:
            return done("bench_press", "lying down with the hands above the chest")
        return done("push_up", "body horizontal with the hands on the floor")
    rowing = (x["torso_lean_median"] >= 30 and x["elbow_range_hinged"] >= 45
              and x["hand_height_median"] < -30)
    if (x["hip_range"] >= 60 and x["torso_lean_range"] >= 40 and x["elbow_range_hinged"] < 30
            and x["torso_lean_median"] < 50   # a row clip spends most of its time hinged over
            and x["hand_height_median"] < -55):
        # Checked before hip drop: from behind or at an angle the hips move back more than down.
        return done("deadlift", "hips and torso hinge down to the floor with the hands hanging")
    hands_on_back = x["hand_height_median"] > -55
    if hands_on_back and x["hip_range"] >= 70 and x["hand_height_median"] < 20 and not rowing:
        # Equipment in front of the hips (a rack's safety bars) can hide how far they drop.
        return done("squat", "hips bend deeply with the hands up at the shoulders")
    if x["hip_drop"] >= 0.28 and (x["hip_range"] >= 45 or x["hip_drop"] >= 0.35) and not rowing:
        if x["hand_height_median"] > -55:
            return done("squat", "hips drop with the hands up at the shoulders")
        if x["torso_lean_range"] >= 40:
            return done("deadlift", "hips and torso hinge down to the floor with the hands hanging")
        return done("squat", "hips drop with the torso staying upright")
    if x["hand_height_median"] > 20:
        if x["elbow_above_shoulder_p10"] > 0.3:
            return done("triceps_extension", "elbows stay up by the head while the forearms move")
        return done("overhead_press", "hands rise overhead from the shoulders")
    if x["side_tilt_range"] >= 15 and x["torso_lean_median"] < 40 and not rowing:
        return done("side_bend", "torso bends side to side")
    if x["torso_lean_median"] >= 18:
        return done("bent_over_row", "elbows bend with the torso hinged forward")
    return done("bicep_curl", "elbows bend with the torso upright and the hands below the shoulders")
