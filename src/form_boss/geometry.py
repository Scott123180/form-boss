"""Per-frame body geometry derived from a PoseTrack. Shared by all lifts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import pose as P
from .pose import PoseTrack


SIDE_VIEW_MIN_YAW = 65.0   # degrees; at or above this the view counts as side-on
FRONT_VIEW_MAX_YAW = 25.0  # at or below this it counts as straight-on (front or back)

JOINTS = {
    "shoulder": (P.L_SHOULDER, P.R_SHOULDER), "elbow": (P.L_ELBOW, P.R_ELBOW),
    "wrist": (P.L_WRIST, P.R_WRIST), "hip": (P.L_HIP, P.R_HIP), "knee": (P.L_KNEE, P.R_KNEE),
    "ankle": (P.L_ANKLE, P.R_ANKLE), "heel": (P.L_HEEL, P.R_HEEL), "toe": (P.L_TOE, P.R_TOE),
}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees, per frame."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _mid(x: np.ndarray, i: int, j: int) -> np.ndarray:
    return (x[:, i] + x[:, j]) / 2


@dataclass
class Geometry:
    """Time series, one value per frame. Angles in degrees, lengths in metres."""

    knee_angle: np.ndarray      # 180 = straight leg
    hip_angle: np.ndarray       # shoulder-hip-knee, 180 = standing tall
    torso_lean: np.ndarray      # torso angle from vertical, 0 = upright
    thigh_angle: np.ndarray     # thigh above horizontal; <= 0 means hip joint at/below knee joint
    hip_below_knee: np.ndarray  # metres the hip joint sits below the knee joint (negative = above)
    bar_offset: np.ndarray      # bar (wrist proxy) ahead of midfoot along the facing direction
    bar_px: np.ndarray          # (T, 2) bar proxy in pixels, for drawing
    hip_px: np.ndarray          # (T, 2) mid-hip in pixels
    femur: float
    tibia: float
    torso: float
    foot: float
    view_yaw: float             # 0 = camera faces front/back of lifter, 90 = true side view
    mode: str = "3d"            # "side-2d": angles measured directly in the image plane
    near: int = 0               # 0 = left side of the body is the better-seen one, 1 = right
    facing: float = 1.0         # side view: +1 if the lifter faces image right, -1 if left

    @property
    def view(self) -> str:
        return ("side" if self.view_yaw >= SIDE_VIEW_MIN_YAW
                else "front" if self.view_yaw <= FRONT_VIEW_MAX_YAW else "oblique")


def compute(t: PoseTrack) -> Geometry:
    w = t.world
    sh, hip = _mid(w, P.L_SHOULDER, P.R_SHOULDER), _mid(w, P.L_HIP, P.R_HIP)
    knee, ankle = _mid(w, P.L_KNEE, P.R_KNEE), _mid(w, P.L_ANKLE, P.R_ANKLE)
    heel, toe = _mid(w, P.L_HEEL, P.R_HEEL), _mid(w, P.L_TOE, P.R_TOE)
    wrist = _mid(w, P.L_WRIST, P.R_WRIST)

    # Weight left/right by visibility so the camera-side limb dominates in a side view.
    def sided(fn) -> np.ndarray:
        vl = t.vis[:, [P.L_HIP, P.L_KNEE, P.L_ANKLE]].mean() + 1e-3
        vr = t.vis[:, [P.R_HIP, P.R_KNEE, P.R_ANKLE]].mean() + 1e-3
        return (fn("L") * vl + fn("R") * vr) / (vl + vr)

    idx = {
        "L": (P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE),
        "R": (P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE),
    }
    knee_angle = sided(lambda s: _angle(w[:, idx[s][1]], w[:, idx[s][2]], w[:, idx[s][3]]))
    hip_angle = sided(lambda s: _angle(w[:, idx[s][0]], w[:, idx[s][1]], w[:, idx[s][2]]))
    femur_ts = sided(lambda s: np.linalg.norm(w[:, idx[s][1]] - w[:, idx[s][2]], axis=-1))
    tibia_ts = sided(lambda s: np.linalg.norm(w[:, idx[s][2]] - w[:, idx[s][3]], axis=-1))

    up = np.array([0.0, -1.0, 0.0])
    torso_vec = sh - hip
    torso_len = np.linalg.norm(torso_vec, axis=-1)
    torso_lean = np.degrees(np.arccos(np.clip((torso_vec @ up) / (torso_len + 1e-9), -1, 1)))

    femur = float(np.median(femur_ts))
    hip_below_knee = hip[:, 1] - knee[:, 1]  # y points down
    thigh_angle = np.degrees(np.arcsin(np.clip(-hip_below_knee / femur, -1, 1)))

    # Facing direction: heel -> toe, flattened to the floor plane, averaged over the clip.
    f = np.median(toe - heel, axis=0)
    f[1] = 0
    f /= np.linalg.norm(f) + 1e-9
    midfoot = (heel + toe) / 2
    bar_offset = (wrist - midfoot) @ f

    # View angle from how much of the hip/shoulder line points along the camera axis.
    def yaw(i: int, j: int) -> float:
        d = np.median(w[:, i] - w[:, j], axis=0)
        return float(np.degrees(np.arctan2(abs(d[2]), abs(d[0]))))

    view_yaw = (yaw(P.L_HIP, P.R_HIP) + yaw(P.L_SHOULDER, P.R_SHOULDER)) / 2
    mode = "3d"
    body = [i for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle") for i in JOINTS[name]]
    near = int(t.vis[:, body[1::2]].mean() > t.vis[:, body[0::2]].mean())
    facing = float(np.sign(np.median(t.image[:, JOINTS["toe"][near], 0] - t.image[:, JOINTS["heel"][near], 0])) or 1.0)
    if view_yaw >= SIDE_VIEW_MIN_YAW:
        # True side view: the sagittal plane is the image plane, so measure there directly instead
        # of trusting the model's depth guess. Use the limb facing the camera.
        mode = "side-2d"
        left = t.vis[:, [P.L_HIP, P.L_KNEE, P.L_ANKLE]].mean() >= t.vis[:, [P.R_HIP, P.R_KNEE, P.R_ANKLE]].mean()
        i_sh, i_hip, i_knee, i_ank, i_heel, i_toe = (
            (P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_HEEL, P.L_TOE) if left
            else (P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_HEEL, P.R_TOE))
        im = t.image
        s2, h2, k2, a2 = im[:, i_sh], im[:, i_hip], im[:, i_knee], im[:, i_ank]
        knee_angle = _angle(h2, k2, a2)
        hip_angle = _angle(s2, h2, k2)
        tv = s2 - h2
        torso_lean = np.degrees(np.arctan2(np.abs(tv[:, 0]), -tv[:, 1]))
        thigh_angle = np.degrees(np.arctan2(k2[:, 1] - h2[:, 1], np.abs(h2[:, 0] - k2[:, 0])))
        m_per_px = femur / np.median(np.linalg.norm(h2 - k2, axis=-1))
        hip_below_knee = (h2[:, 1] - k2[:, 1]) * m_per_px
        facing = np.sign(np.median(im[:, i_toe, 0] - im[:, i_heel, 0])) or 1.0
        wrist_x = _mid(im, P.L_WRIST, P.R_WRIST)[:, 0]
        midfoot_x = (im[:, i_heel, 0] + im[:, i_toe, 0]) / 2
        bar_offset = facing * (wrist_x - midfoot_x) * m_per_px

    return Geometry(
        knee_angle=knee_angle,
        hip_angle=hip_angle,
        torso_lean=torso_lean,
        thigh_angle=thigh_angle,
        hip_below_knee=hip_below_knee,
        bar_offset=bar_offset,
        bar_px=_mid(t.image, P.L_WRIST, P.R_WRIST),
        hip_px=_mid(t.image, P.L_HIP, P.R_HIP),
        femur=femur,
        tibia=float(np.median(tibia_ts)),
        torso=float(np.median(torso_len)),
        foot=float(np.median(np.linalg.norm(toe - heel, axis=-1))),
        view_yaw=view_yaw,
        mode=mode,
        near=near,
        facing=facing,
    )


# ---- Measurements lifts build their signals from -------------------------------------------------
#
# A joint angle is only readable straight off the image when the joint moves in the image plane:
# a curl seen from the side, a press seen from the front. `plane` names the body plane the movement
# lives in; when the camera view matches it the angle is measured in the image, otherwise from the
# model's 3D estimate, which is less exact.

_PLANE_VIEW = {"sagittal": "side", "frontal": "front"}


def in_plane(g: Geometry, plane: str) -> bool:
    return g.view == _PLANE_VIEW[plane]


def _pts(t: PoseTrack, g: Geometry, plane: str) -> np.ndarray:
    return t.image if in_plane(g, plane) else t.world


def joint_angle(t: PoseTrack, g: Geometry, a: str, b: str, c: str, side: int, plane: str) -> np.ndarray:
    """Angle at joint `b` between `a` and `c` on one side of the body (0 = left, 1 = right)."""
    x = _pts(t, g, plane)
    return _angle(x[:, JOINTS[a][side]], x[:, JOINTS[b][side]], x[:, JOINTS[c][side]])


def image_angle(t: PoseTrack, a: str, b: str, c: str, side: int) -> np.ndarray:
    """Angle at joint `b` as it appears in the image, whatever the view. Only trustworthy for
    telling straight from bent: a straight limb looks straight from any angle, a bent one does not
    look as bent as it is unless it bends in the image plane."""
    x = t.image
    return _angle(x[:, JOINTS[a][side]], x[:, JOINTS[b][side]], x[:, JOINTS[c][side]])


def elbow_angle(t: PoseTrack, g: Geometry, side: int, plane: str) -> np.ndarray:
    """180 = straight arm."""
    return joint_angle(t, g, "shoulder", "elbow", "wrist", side, plane)


def arm_angle(t: PoseTrack, g: Geometry, side: int, plane: str) -> np.ndarray:
    """Upper arm relative to the torso: 0 = along the ribs, 180 = straight overhead.

    From a side view the angle is signed: negative when the elbow is behind the torso line.
    """
    a = joint_angle(t, g, "hip", "shoulder", "elbow", side, plane)
    if plane == "sagittal" and in_plane(g, plane):
        im = t.image
        sh = im[:, JOINTS["shoulder"][side]]
        u, v = im[:, JOINTS["hip"][side]] - sh, im[:, JOINTS["elbow"][side]] - sh
        cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]   # image y points down
        a = a * np.where(cross * g.facing > 0, -1.0, 1.0)
    return a


def tilt(t: PoseTrack, a: str, b: str, side: int) -> np.ndarray:
    """Angle of the segment a -> b from straight up, in the image. Signed: + = leaning image right."""
    d = t.image[:, JOINTS[b][side]] - t.image[:, JOINTS[a][side]]
    return np.degrees(np.arctan2(d[:, 0], -d[:, 1]))


def tilt_mid(t: PoseTrack, a: str, b: str) -> np.ndarray:
    """Like `tilt`, between the left/right midpoints of the two joints."""
    d = mid(t, b) - mid(t, a)
    return np.degrees(np.arctan2(d[:, 0], -d[:, 1]))


def mid(t: PoseTrack, joint: str) -> np.ndarray:
    """(T, 2) midpoint of the left and right `joint` in pixels."""
    return _mid(t.image, *JOINTS[joint])


def hand_height(t: PoseTrack, side: int) -> np.ndarray:
    """Wrist height above the shoulder as % of arm length: -100 hanging, 0 level, 100 overhead."""
    im = t.image
    sh, el, wr = (im[:, JOINTS[j][side]] for j in ("shoulder", "elbow", "wrist"))
    arm = np.linalg.norm(sh - el, axis=-1) + np.linalg.norm(el - wr, axis=-1)
    return 100 * (sh[:, 1] - wr[:, 1]) / np.percentile(arm, 95)   # foreshortening only shortens
