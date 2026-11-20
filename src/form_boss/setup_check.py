"""Camera setup checks: tell the user how to re-shoot when the footage limits the analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import pose as P
from .geometry import JOINTS, Geometry
from .lifts.base import Lift
from .pose import PoseTrack


@dataclass
class SetupFinding:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def check(t: PoseTrack, g: Geometry, lift: Lift) -> list[SetupFinding]:
    out: list[SetupFinding] = []

    # View angle
    yaw = g.view_yaw
    ok = g.view == lift.view
    if not ok and g.view in lift.also_views:
        out.append(SetupFinding(
            "view angle", True, f"filmed from the {'front or back' if g.view == 'front' else g.view} "
            f"(about {yaw:.0f} deg off straight-on): the checks for this view are judged; film from "
            f"the {lift.view} for the rest"))
    elif lift.view == "side":
        out.append(SetupFinding(
            "view angle", ok,
            f"camera is about {90 - yaw:.0f} deg off a true side view",
            "" if ok else (lift.camera or "Put the camera level with your hips, pointing straight "
            "at your side (perpendicular to the direction you face).") + " Measurements along the "
            "direction you face are unreliable until then.",
        ))
    else:
        out.append(SetupFinding(
            "view angle", ok, f"camera is about {yaw:.0f} deg off a straight-on view",
            "" if ok else "Put the camera directly in front of you at chest height. Side-to-side "
            "measurements are unreliable until then.",
        ))

    # Camera tilt: rolled sideways (measurements are corrected for it) or looking up from the floor.
    # Pitch is the pose model's estimate and reads low, so a modest reading already means a lot.
    if g.camera_pitch > 8 or g.camera_pitch < -15 or abs(g.camera_roll) >= 3:
        parts, fix = [], []
        if g.camera_pitch > 8:
            parts.append("looking up at you from low down")
            fix.append("Raise the camera to hip height and point it level. From the floor, equipment "
                       "between you and the camera (a rack's safety bars, plates) hides the hips at "
                       "the bottom, and the tracker guesses them too high.")
        elif g.camera_pitch < -15:
            parts.append("looking down at you from above")
            fix.append("Lower the camera to hip height and point it level.")
        if abs(g.camera_roll) >= 3:
            parts.append(f"tilted {abs(g.camera_roll):.0f} deg sideways (measurements are corrected for this)")
            fix.append("Stand the phone square to the floor.")
        out.append(SetupFinding("camera tilt", False, "camera is " + " and ".join(parts), " ".join(fix)))
    elif g.corrected or g.camera_pitch:
        out.append(SetupFinding("camera tilt", True, "camera is about level"))

    # Whole body in frame
    pts = t.image[:, [P.NOSE, P.L_HEEL, P.R_HEEL, P.L_TOE, P.R_TOE, P.L_WRIST, P.R_WRIST]]
    m = 0.02
    inside = (
        (pts[..., 0] > t.width * m) & (pts[..., 0] < t.width * (1 - m))
        & (pts[..., 1] > t.height * m) & (pts[..., 1] < t.height * (1 - m))
    ).all(axis=1)
    frac = float(inside.mean())
    out.append(SetupFinding(
        "framing", frac > 0.95, f"head, hands and feet in frame for {frac:.0%} of the clip",
        "" if frac > 0.95 else "Move the camera back until your whole body, including feet, "
        "stays in frame for the entire rep.",
    ))

    # Subject size: head to heels, measured along whichever way the body is lying
    ends = t.image[:, [P.NOSE, P.L_HEEL, P.R_HEEL]]
    tall = np.percentile(np.ptp(ends[..., 1], axis=1), 95) / t.height
    wide = np.percentile(np.ptp(ends[..., 0], axis=1), 95) / t.width
    size = float(max(tall, wide))
    ok = 0.35 <= size
    out.append(SetupFinding(
        "subject size", ok, f"you fill {size:.0%} of the frame {'height' if tall >= wide else 'width'}",
        "" if ok else "Move the camera closer; aim to fill half to three quarters of the frame.",
    ))

    # Visibility of the joints this lift is judged on (plates, racks and spotters hide them).
    # From the side only the near limb can be seen; from the front both sides should be.
    left = t.vis[:, [JOINTS[j][0] for j in lift.joints]].mean(axis=1)
    right = t.vis[:, [JOINTS[j][1] for j in lift.joints]].mean(axis=1)
    v = float((np.minimum(left, right) if g.view == "front" else np.maximum(left, right)).mean())
    names = "/".join(lift.joints)
    out.append(SetupFinding(
        "joint visibility", v > 0.7, f"{names} tracking confidence {v:.0%}",
        "" if v > 0.7 else f"Something is hiding your {names} (equipment, other people, clothing "
        "against a similar background). Raise or shift the camera so they stay visible.",
    ))

    # Detection gaps and frame rate
    det = float(t.found.mean())
    out.append(SetupFinding(
        "tracking", det > 0.9, f"lifter tracked reliably in {det:.0%} of frames",
        "" if det > 0.9 else "The tracker kept losing you. Improve lighting, wear clothes that contrast "
        "with the background, keep your head visible and keep other people out of the frame.",
    ))
    src = t.source_fps or t.fps
    out.append(SetupFinding(
        "frame rate", src >= 24, f"{src:.0f} fps" + (f", analysed at {t.fps:.0f}" if t.stride > 1 else ""),
        "" if src >= 24 else "Record at 30 fps or higher.",
    ))
    if t.others / max(1, len(t)) > 0.02:
        share = t.others / len(t)
        out.append(SetupFinding(
            "other people", share < 0.1, f"the tracker followed someone else for {share:.0%} of the clip; "
            "those frames were set aside",
            "" if share < 0.1 else "Film where nobody walks between you and the camera, or stand closer "
            "to it than anyone else in the picture.",
        ))
    return out
