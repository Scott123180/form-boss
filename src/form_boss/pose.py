"""Pose tracking: video -> per-frame landmarks (MediaPipe Pose Landmarker).

Tracking runs in two steps. `detect` runs the pose model over the video and keeps every person it
finds in each frame (up to MAX_POSES). `build_track` then works out which of them is the lifter and
cleans up that one skeleton. Detection is the slow part, so `track` can cache its output.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .errors import TrackingError, VideoError

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_heavy.task"

# People tracked per frame. One: with more slots MediaPipe re-runs person detection on every frame
# instead of following the lifter from the last one, and the landmarks get about 8x noisier (the
# knee flips at the bottom of a squat). A bystander who takes over the single slot when the lifter
# is lost is caught afterwards by `pick_lifter`, which sees the skeleton jump.
MAX_POSES = 1
# High frame rates are analysed at about this rate: movement over ~33 ms is far below what the
# smoothing keeps, so 60-240 fps footage only costs time.
TARGET_FPS = 30.0
CACHE_VERSION = 1   # bump when detect() output changes meaning

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
    """Landmarks for every analysed frame of a video.

    image: (T, 33, 2) pixel coordinates.
    world: (T, 33, 3) metres, origin at mid-hip, x = image right, y = down, z = away from camera.
    vis:   (T, 33) landmark visibility 0..1.
    found: (T,) whether the lifter was tracked plausibly in the frame (gaps are interpolated).
    fps:   rate of the analysed frames. Analysed frame i is video frame i * stride.
    """

    image: np.ndarray
    world: np.ndarray
    vis: np.ndarray
    found: np.ndarray
    fps: float
    width: int
    height: int
    stride: int = 1
    source_fps: float = 0.0
    others: int = 0   # frames where someone other than the lifter was tracked and set aside

    def __len__(self) -> int:
        return len(self.image)


@dataclass
class Detections:
    """Everyone the pose model found, per analysed frame, before the lifter is picked.

    Arrays are padded to `max_poses` people; `count` says how many slots of each frame are filled.
    The model's order of people is not stable from frame to frame, and with one slot the person in
    it can change when the model loses the lifter.
    """

    image: np.ndarray   # (T, K, 33, 2)
    world: np.ndarray   # (T, K, 33, 3)
    vis: np.ndarray     # (T, K, 33)
    count: np.ndarray   # (T,)
    fps: float
    width: int
    height: int
    stride: int
    source_fps: float
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, image=self.image.astype(np.float32), world=self.world.astype(np.float32),
            vis=self.vis.astype(np.float16), count=self.count,
            info=np.array([self.fps, self.width, self.height, self.stride, self.source_fps]),
            key=np.array(_cache_key_str(self.meta)))

    @classmethod
    def load(cls, path: Path) -> Detections:
        z = np.load(path)
        fps, w, h, stride, src = z["info"]
        return cls(z["image"].astype(float), z["world"].astype(float), z["vis"].astype(float),
                   z["count"], float(fps), int(w), int(h), int(stride), float(src))


def _ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading pose model to %s", MODEL_PATH)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def frame_stride(source_fps: float, target_fps: float = TARGET_FPS) -> int:
    """Analyse every n-th frame so the analysed rate stays at or above `target_fps`: 60 fps -> 2,
    120 -> 4, 240 -> 8. Rates under twice the target (24-59 fps) are kept as they are."""
    return max(1, int(source_fps // target_fps))


def _cache_key(video: Path, max_poses: int, target_fps: float) -> dict:
    st = video.stat()
    return {"v": CACHE_VERSION, "size": st.st_size, "mtime": int(st.st_mtime), "poses": max_poses,
            "fps": target_fps, "model": MODEL_PATH.name}


def _cache_key_str(meta: dict) -> str:
    return ";".join(f"{k}={meta[k]}" for k in sorted(meta))


def detect(video: Path, max_poses: int = MAX_POSES, target_fps: float = TARGET_FPS) -> Detections:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise VideoError(f"Cannot open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = frame_stride(src_fps, target_fps)
    fps = src_fps / stride
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Tracking %s: %d frames at %.0f fps%s", video.name, total, src_fps,
             f", analysing 1 frame in {stride} ({fps:.0f} fps)" if stride > 1 else "")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=max_poses,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    image, world, vis, count = [], [], [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * stride * 1000 / src_fps))
            im = np.zeros((max_poses, 33, 2))
            wo = np.zeros((max_poses, 33, 3))
            vi = np.zeros((max_poses, 33))
            for k, (lm, wl) in enumerate(zip(res.pose_landmarks, res.pose_world_landmarks)):
                im[k] = [(p.x * w, p.y * h) for p in lm]
                wo[k] = [(p.x, p.y, p.z) for p in wl]
                vi[k] = [p.visibility for p in lm]
            image.append(im)
            world.append(wo)
            vis.append(vi)
            count.append(len(res.pose_landmarks))
            i += 1
            if i % 500 == 0:
                log.debug("  %d frames", i)
            for _ in range(stride - 1):   # skip without converting
                if not cap.grab():
                    break
    cap.release()
    if not image:
        raise VideoError(f"No frames could be read from {video}")
    return Detections(np.array(image), np.array(world), np.array(vis), np.array(count), fps, w, h,
                      stride, src_fps)


# ---- Picking the lifter -------------------------------------------------------------------------

def _body_size(image: np.ndarray) -> np.ndarray:
    """Shoulder-hip-knee-ankle chain length in pixels, the longer side. Any leading shape (..., 33, 2)."""
    def chain(sh, hip, knee, ankle):
        pts = image[..., [sh, hip, knee, ankle], :]
        return np.linalg.norm(np.diff(pts, axis=-2), axis=-1).sum(axis=-1)
    return np.maximum(chain(L_SHOULDER, L_HIP, L_KNEE, L_ANKLE), chain(R_SHOULDER, R_HIP, R_KNEE, R_ANKLE))


def _centre(image: np.ndarray) -> np.ndarray:
    return image[..., [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :].mean(axis=-2)


def tracklets(det: Detections, max_gap_s: float = 0.5) -> list[dict]:
    """Link the people found in consecutive frames into tracklets by position.

    A person is joined to the tracklet whose last position is nearest, if within a quarter of a
    body length (more across a gap of a few frames). Returns [{"t": frames, "k": slots}, ...].
    """
    centre, size = _centre(det.image), _body_size(det.image)
    max_gap = max(1, int(max_gap_s * det.fps))
    tracks: list[dict] = []
    for t in range(len(det.count)):
        n = int(det.count[t])
        if not n:
            continue
        live = [i for i, tr in enumerate(tracks) if t - tr["t"][-1] <= max_gap]
        pairs = []
        for i in live:
            tr = tracks[i]
            last_t, last_k = tr["t"][-1], tr["k"][-1]
            ref = size[last_t, last_k] + 1e-9
            allow = 0.25 + 0.5 * (t - last_t - 1) / det.fps   # metres-ish of slack per second of gap
            for k in range(n):
                d = np.linalg.norm(centre[t, k] - centre[last_t, last_k]) / ref
                if d < allow and 0.6 < size[t, k] / ref < 1.6:
                    pairs.append((d, i, k))
        used_tr, used_k = set(), set()
        for _, i, k in sorted(pairs):
            if i in used_tr or k in used_k:
                continue
            tracks[i]["t"].append(t)
            tracks[i]["k"].append(k)
            used_tr.add(i)
            used_k.add(k)
        for k in range(n):
            if k not in used_k:
                tracks.append({"t": [t], "k": [k]})
    return tracks


def pick_lifter(det: Detections) -> tuple[np.ndarray, np.ndarray]:
    """Slot of the lifter in each frame (-1 where not found) and where someone else was set aside.

    The tracked skeleton is split wherever it jumps (see `tracklets`). The lifter is the person who
    fills the most of the picture for the most of the clip: the tracklet with the largest summed
    body size. Other tracklets are added to the lifter's timeline where it has gaps, if they stand
    where the lifter stands and are about the lifter's size; a bystander walking past, or the
    tracker latching onto someone across the gym, fails one of those.
    """
    T = len(det.count)
    slot = np.full(T, -1)
    tracks = tracklets(det)
    if not tracks:
        return slot, np.zeros(T, bool)
    size, centre = _body_size(det.image), _centre(det.image)
    for tr in tracks:
        s = size[tr["t"], tr["k"]]
        tr["score"] = float(s.sum())
        tr["size"] = float(np.median(s))
        tr["centre"] = np.median(centre[tr["t"], tr["k"]], axis=0)
    tracks.sort(key=lambda tr: -tr["score"])
    best = tracks[0]
    slot[best["t"]] = best["k"]
    for tr in tracks[1:]:
        t = np.array(tr["t"])
        if (slot[t] >= 0).mean() > 0.1:
            continue
        ok_size = 0.7 < tr["size"] / best["size"] < 1.4
        ok_place = np.linalg.norm(tr["centre"] - best["centre"]) < 0.8 * best["size"]
        if ok_size and ok_place:
            free = slot[t] < 0
            slot[t[free]] = np.array(tr["k"])[free]
    others = (det.count > (slot >= 0)).astype(bool)
    return slot, others


# ---- Cleaning up the lifter's skeleton ----------------------------------------------------------

def _plausible(image: np.ndarray, found: np.ndarray) -> np.ndarray:
    """Reject frames where the skeleton is clearly not the lifter.

    The tracker sometimes locks onto a patch of clothing or equipment. Those frames show up as a
    skeleton of the wrong size, one that is upside down, or one that teleports.
    """
    size = _body_size(image)
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


def build_track(det: Detections) -> PoseTrack:
    slot, others = pick_lifter(det)
    found = slot >= 0
    if found.sum() < 10:
        raise TrackingError("No person detected in the video.")
    T = len(slot)
    k = np.maximum(slot, 0)
    image, world, vis = (a[np.arange(T), k] for a in (det.image, det.world, det.vis))
    vis = np.where(found[:, None], vis, 0.0)
    found &= _plausible(image, found)
    if found.sum() < 10:
        raise TrackingError("Could not track the lifter reliably in this video.")
    fps = det.fps
    image = _despike(_fill_gaps(image, found), fps)
    world = _despike(_fill_gaps(world, found), fps)
    return PoseTrack(
        image=_smooth(image, fps, 0.2),
        world=_smooth(world, fps, 0.35),
        vis=vis,
        found=found,
        fps=fps,
        width=det.width,
        height=det.height,
        stride=det.stride,
        source_fps=det.source_fps,
        others=int(others.sum()),
    )


def detections(video: Path, cache: Path | None = None, max_poses: int = MAX_POSES,
               target_fps: float = TARGET_FPS) -> Detections:
    """Run `detect`, or load its output from `cache` if that was made from this same video file."""
    if not video.exists():
        raise VideoError(f"No such video: {video}")
    key = _cache_key(video, max_poses, target_fps)
    if cache is not None and cache.exists():
        try:
            if str(np.load(cache)["key"]) == _cache_key_str(key):
                log.info("Using cached pose tracking from %s", cache)
                return Detections.load(cache)
        except (OSError, KeyError, ValueError):
            pass
        log.info("Cached tracking in %s is from a different video or settings; tracking again", cache)
    det = detect(video, max_poses, target_fps)
    det.meta = key
    if cache is not None:
        det.save(cache)
    return det


def track(video: Path, cache: Path | None = None) -> PoseTrack:
    return build_track(detections(video, cache))
