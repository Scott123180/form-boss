"""Per-frame body geometry derived from a PoseTrack. Shared by all lifts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import percentile_filter

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
    camera_pitch: float = 0.0   # degrees the camera looks up (+) or down (-), from the standing lifter
    camera_roll: float = 0.0    # degrees "up" in the picture is turned from true vertical at the lifter
    corrected: str = ""         # how measurements were corrected for the camera tilt, if at all

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


# ---- Camera tilt --------------------------------------------------------------------------------
#
# A standing lifter's leg is vertical (the median lean on the Mendeley side views is 0.6 deg), so it
# shows which way is up at the lifter: in the picture, where it gives the camera's sideways roll
# plus the keystone lean that looking up or down adds away from the picture's centre, and in the
# pose model's 3D estimate, where it also shows the camera looking up or down (pitch).
#
# The 3D estimate's pitch is rough: on the user's floor-level clips, filmed looking ~30 deg up, it
# reads ~11 deg. It is only used to decide when 3D is worth falling back on.

ROLL_MIN_DEG = 2.0    # less than this is within tracking noise and natural lean: left alone
ROLL_MAX_DEG = 15.0   # more than this and the "standing" frames are probably not upright
PITCH_3D_DEG = 10.0   # camera pitched further than this: thigh angle and depth from levelled 3D


def standing(t: PoseTrack, g: Geometry) -> np.ndarray:
    """(T,) frames where the lifter stands tall: knee and hip straight, leg near vertical."""
    sides = (g.near,) if g.view == "side" else (0, 1)
    knee = np.min([image_angle(t, "hip", "knee", "ankle", s) for s in sides], axis=0)
    hip = np.min([image_angle(t, "shoulder", "hip", "knee", s) for s in sides], axis=0)
    return (knee > 168) & (hip > 165) & (np.abs(_leg_tilt(t, g)) < 20) & t.found


def _leg_tilt(t: PoseTrack, g: Geometry) -> np.ndarray:
    """Ankle-to-hip direction in the picture, degrees from straight up, + = top leaning right."""
    if g.view == "side":
        d = t.image[:, JOINTS["hip"][g.near]] - t.image[:, JOINTS["ankle"][g.near]]
    else:
        d = mid(t, "hip") - mid(t, "ankle")
    return np.degrees(np.arctan2(d[:, 0], -d[:, 1]))


def camera_tilt(t: PoseTrack, g: Geometry) -> tuple[float, float, np.ndarray | None]:
    """(roll, pitch, up): picture tilt at the lifter (+ = up leans right), camera pitch (+ = looking
    up), and the 3D up direction, from the frames where the lifter stands tall. Zeros and None
    when there are fewer than half a second of them (bench, push-up, a clip that starts mid-set)."""
    st = standing(t, g)
    if st.sum() < max(5, int(0.5 * t.fps)):
        return 0.0, 0.0, None
    roll = float(np.median(_leg_tilt(t, g)[st]))
    if g.view == "side":
        d = t.world[:, JOINTS["hip"][g.near]] - t.world[:, JOINTS["ankle"][g.near]]
    else:
        d = _mid(t.world, *JOINTS["hip"]) - _mid(t.world, *JOINTS["ankle"])
    d = d[st] / (np.linalg.norm(d[st], axis=-1, keepdims=True) + 1e-9)
    up = np.median(d, axis=0)
    up /= np.linalg.norm(up) + 1e-9
    pitch = float(np.degrees(np.arctan2(up[2], -up[1])))   # top further from the camera: looking up
    return roll, pitch, up


def _rotation_to_up(up: np.ndarray) -> np.ndarray:
    """Rotation matrix taking `up` to (0, -1, 0), straight up in the pose model's y-down axes."""
    target = np.array([0.0, -1.0, 0.0])
    v, c = np.cross(up, target), float(up @ target)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s ** 2)


def _level_image(t: PoseTrack, roll: float) -> np.ndarray:
    """Image coordinates turned by -roll about the lifter's hips, so standing reads vertical."""
    a = np.radians(-roll)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.median(mid(t, "hip")[t.found], axis=0)
    return (t.image - c) @ rot.T + c


def calibrate(t: PoseTrack) -> tuple[PoseTrack, Geometry]:
    """Geometry corrected for camera tilt, and the track to take further measurements from.

    - The 3D estimate is turned so the standing leg points straight up, making "vertical" and
      "horizontal" in every 3D measurement true ones rather than the camera's.
    - The picture is turned by the camera's roll at the lifter (2-15 deg), for the measurements
      taken against the picture's vertical: torso lean, thigh angle, bar over midfoot. Joint angles
      do not change. The returned track is for measuring only; draw with the original.
    - From the side with the camera pitched more than PITCH_3D_DEG, thigh angle and depth come from
      the levelled 3D estimate: looking up or down at an angle makes level lines that are not square
      to the camera slope in the picture.
    """
    g0 = compute(t)
    roll, pitch, up = camera_tilt(t, g0)
    notes = []
    tm = t
    if up is not None:
        tm = replace(tm, world=t.world @ _rotation_to_up(up).T)
    if ROLL_MIN_DEG <= abs(roll) <= ROLL_MAX_DEG:
        tm = replace(tm, image=_level_image(t, roll))
        notes.append(f"picture levelled by {abs(roll):.0f} deg")
    g = compute(tm)
    if g.mode == "side-2d" and up is not None and abs(pitch) > PITCH_3D_DEG:
        hip, knee = tm.world[:, JOINTS["hip"][g.near]], tm.world[:, JOINTS["knee"][g.near]]
        v = knee - hip
        g.thigh_angle = np.degrees(np.arcsin(np.clip(v[:, 1] / (np.linalg.norm(v, axis=-1) + 1e-9), -1, 1)))
        g.hip_below_knee = hip[:, 1] - knee[:, 1]
        notes.append("thigh angle and depth from the levelled 3D estimate (camera pitched)")
    g.camera_roll, g.camera_pitch, g.corrected = roll, pitch, "; ".join(notes)
    return tm, g


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


def in_frame(t: PoseTrack, joints: tuple[str, ...], margin: float = 0.01) -> np.ndarray:
    """(T,) whether every joint in `joints` is inside the picture on at least one side of the body.
    The model still places joints it cannot see, but those are guesses."""
    x, y = t.image[..., 0], t.image[..., 1]
    inside = ((x > margin * t.width) & (x < (1 - margin) * t.width)
              & (y > margin * t.height) & (y < (1 - margin) * t.height))
    return np.all([inside[:, JOINTS[j][0]] | inside[:, JOINTS[j][1]] for j in joints], axis=0)


# Phone main cameras film video with a ~65-70 deg field of view across the long side of the frame.
FOCAL_X_LONG_SIDE = 0.8


def width_at_hips(t: PoseTrack, joint: str) -> np.ndarray:
    """Left-to-right distance between the two `joint`s in pixels, rescaled to how it would look at
    the hips' distance from the camera. From the front, knees pushed forward at the bottom of a
    squat are closer to the camera than the ankles and look wider apart than they are (and from
    behind, narrower). Uses the 3D estimate's depth offsets and a camera distance worked out from
    shoulder width. For front and rear views."""
    im, w = t.image, t.world
    f = FOCAL_X_LONG_SIDE * max(t.width, t.height)
    sh_px = np.abs(im[:, JOINTS["shoulder"][0], 0] - im[:, JOINTS["shoulder"][1], 0])
    sh_m = np.linalg.norm(w[:, JOINTS["shoulder"][0]] - w[:, JOINTS["shoulder"][1]], axis=-1)
    dist = np.median(f * sh_m[t.found] / np.maximum(sh_px[t.found], 1.0))   # camera to hips, metres
    z = _mid(w, *JOINTS[joint])[:, 2]   # depth of the joint pair relative to the hips, + = further
    px = np.abs(im[:, JOINTS[joint][0], 0] - im[:, JOINTS[joint][1], 0])
    return px * (dist + z) / dist


def mid(t: PoseTrack, joint: str) -> np.ndarray:
    """(T, 2) midpoint of the left and right `joint` in pixels."""
    return _mid(t.image, *JOINTS[joint])


def hand_height(t: PoseTrack, side: int) -> np.ndarray:
    """Wrist height above the shoulder as % of arm length: -100 hanging, 0 level, 100 overhead."""
    im = t.image
    sh, el, wr = (im[:, JOINTS[j][side]] for j in ("shoulder", "elbow", "wrist"))
    arm = np.linalg.norm(sh - el, axis=-1) + np.linalg.norm(el - wr, axis=-1)
    return 100 * (sh[:, 1] - wr[:, 1]) / np.percentile(arm, 95)   # foreshortening only shortens


def relative_to_standing(x: np.ndarray, fps: float, window_s: float = 8.0) -> np.ndarray:
    """`x` as a share of its standing (high) level over the surrounding `window_s` seconds.

    For lengths in the picture that a rep shortens (hip height above the ankles): the lifter's size
    in the picture drifts as they walk toward or away from the camera, so the standing level is
    taken locally, the 90th percentile of the window, which reps take only a small part of.
    """
    size = max(3, int(window_s * fps) | 1)
    top = percentile_filter(x, 90, size=size, mode="nearest") if len(x) > size else np.full_like(x, np.percentile(x, 90))
    return x / (top + 1e-9)
