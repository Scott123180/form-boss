"""Pose tracking: video -> per-frame landmarks (MediaPipe Pose Landmarker)."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_heavy.task"

# MediaPipe landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_ANKLE, L_HEEL), (L_HEEL, L_TOE), (L_ANKLE, L_TOE),
    (R_ANKLE, R_HEEL), (R_HEEL, R_TOE), (R_ANKLE, R_TOE),
]


@dataclass
class PoseTrack:
    """Landmarks for every frame of a video.

    image: (T, 33, 2) pixel coordinates.
    world: (T, 33, 3) metres, origin at mid-hip, x = image right, y = down, z = away from camera.
    vis:   (T, 33) landmark visibility 0..1.
    found: (T,) whether the lifter was tracked plausibly in the frame (gaps are interpolated).
    """

    image: np.ndarray
    world: np.ndarray
    vis: np.ndarray
    found: np.ndarray
    fps: float
    width: int
    height: int

    def __len__(self) -> int:
        return len(self.image)


def _ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _plausible(image: np.ndarray, found: np.ndarray) -> np.ndarray:
    """Reject frames where the skeleton is clearly not the lifter.

    The tracker sometimes locks onto a bystander or a patch of clothing. Those frames show up as a
    skeleton of the wrong size, one that is upside down, or one that teleports.
    """
    def chain(sh, hip, knee, ankle):
        pts = image[:, [sh, hip, knee, ankle]]
        return np.linalg.norm(np.diff(pts, axis=1), axis=-1).sum(axis=1)

    size = np.maximum(chain(L_SHOULDER, L_HIP, L_KNEE, L_ANKLE), chain(R_SHOULDER, R_HIP, R_KNEE, R_ANKLE))
    typical = np.median(size[found])
    hip = (image[:, L_HIP] + image[:, R_HIP]) / 2
    sh = (image[:, L_SHOULDER] + image[:, R_SHOULDER]) / 2
    ok = (size > 0.5 * typical) & (size < 1.6 * typical)
    ok &= sh[:, 1] < hip[:, 1] + 0.2 * typical           # not upside down (a deep row hinge still passes)
    ok &= np.linalg.norm(hip - np.median(hip[found & ok], axis=0), axis=-1) < 0.8 * typical  # stays put
    return ok


def _fill_gaps(a: np.ndarray, found: np.ndarray) -> np.ndarray:
    idx = np.arange(len(a))
    flat = a.reshape(len(a), -1).copy()
    for c in range(flat.shape[1]):
        flat[~found, c] = np.interp(idx[~found], idx[found], flat[found, c])
    return flat.reshape(a.shape)


def _despike(a: np.ndarray, fps: float) -> np.ndarray:
    """Median filter over ~0.2 s: removes single-frame jumps where a limb snaps to the wrong place."""
    return median_filter(a, size=(max(3, int(fps * 0.2) | 1), 1, 1), mode="nearest")


def _smooth(a: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    win = int(fps * seconds) | 1
    if win < 5 or len(a) <= win:
        return a
    return savgol_filter(a, win, 2, axis=0)


def track(video: Path) -> PoseTrack:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    image, world, vis, found = [], [], [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps)
            )
            i += 1
            if res.pose_landmarks:
                lm, wl = res.pose_landmarks[0], res.pose_world_landmarks[0]
                image.append([(p.x * w, p.y * h) for p in lm])
                world.append([(p.x, p.y, p.z) for p in wl])
                vis.append([p.visibility for p in lm])
                found.append(True)
            else:
                image.append(np.zeros((33, 2)))
                world.append(np.zeros((33, 3)))
                vis.append(np.zeros(33))
                found.append(False)
    cap.release()

    found_a = np.array(found)
    if found_a.sum() < 10:
        raise SystemExit("No person detected in the video.")
    image_a = np.array(image, dtype=float)
    found_a &= _plausible(image_a, found_a)
    if found_a.sum() < 10:
        raise SystemExit("Could not track the lifter reliably in this video.")
    image_a = _despike(_fill_gaps(image_a, found_a), fps)
    world_a = _despike(_fill_gaps(np.array(world, dtype=float), found_a), fps)
    return PoseTrack(
        image=_smooth(image_a, fps, 0.2),
        world=_smooth(world_a, fps, 0.35),
        vis=np.array(vis, dtype=float),
        found=found_a,
        fps=fps,
        width=w,
        height=h,
    )
