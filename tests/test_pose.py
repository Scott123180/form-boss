"""Lifter selection, frame stride and the detection cache, on synthetic skeletons."""

import numpy as np

from form_boss import pose as P

FPS = 30.0


def skeleton(x: float, y: float = 500.0, scale: float = 1.0) -> np.ndarray:
    """A standing figure: (33, 2) pixels, feet at `y`, centred on `x`, ~600 px tall at scale 1."""
    s = np.zeros((33, 2))
    h = 600 * scale
    pts = {P.NOSE: 1.0, P.L_SHOULDER: 0.82, P.R_SHOULDER: 0.82, P.L_ELBOW: 0.62, P.R_ELBOW: 0.62,
           P.L_WRIST: 0.45, P.R_WRIST: 0.45, P.L_HIP: 0.52, P.R_HIP: 0.52, P.L_KNEE: 0.28,
           P.R_KNEE: 0.28, P.L_ANKLE: 0.04, P.R_ANKLE: 0.04, P.L_HEEL: 0.0, P.R_HEEL: 0.0,
           P.L_TOE: 0.0, P.R_TOE: 0.0}
    for j, frac in pts.items():
        side = -1 if j % 2 else 1
        s[j] = (x + side * 20 * scale, y - frac * h)
    s[P.L_TOE, 0] += 40 * scale
    s[P.R_TOE, 0] += 40 * scale
    return s


def detections(frames: list[np.ndarray | None]) -> P.Detections:
    T = len(frames)
    image = np.zeros((T, 1, 33, 2))
    count = np.zeros(T, int)
    for t, sk in enumerate(frames):
        if sk is not None:
            image[t, 0] = sk
            count[t] = 1
    world = np.zeros((T, 1, 33, 3))
    vis = np.where(count[:, None, None] > 0, 0.9, 0.0) * np.ones((T, 1, 33))
    return P.Detections(image, world, vis, count, FPS, 1080, 1920, 1, FPS)


def test_frame_stride():
    assert [P.frame_stride(f) for f in (15, 24, 30, 50, 59.94, 60, 120, 240)] == [1, 1, 1, 1, 1, 2, 4, 8]


def test_bystander_taking_the_slot_is_dropped():
    # The tracker loses the lifter for a second and latches onto a smaller person across the gym.
    lifter, other = skeleton(500), skeleton(1500, 400, 0.5)
    frames = [lifter] * 60 + [other] * 30 + [lifter] * 60
    slot, others = P.pick_lifter(detections(frames))
    assert (slot[:60] == 0).all() and (slot[90:] == 0).all()
    assert (slot[60:90] == -1).all() and others[60:90].all() and not others[:60].any()


def test_lifter_reacquired_after_a_gap_is_kept():
    # Lost for half a second (no detection), found again in the same place.
    lifter = skeleton(500)
    frames = [lifter] * 60 + [None] * 15 + [skeleton(510)] * 60
    slot, _ = P.pick_lifter(detections(frames))
    assert (slot[:60] == 0).all() and (slot[75:] == 0).all()


def test_lifter_who_walks_in_is_one_tracklet():
    frames = [skeleton(200 + 5 * i) for i in range(60)] + [skeleton(500)] * 90
    tr = P.tracklets(detections(frames))
    assert len(tr) == 1 and len(tr[0]["t"]) == 150


def test_build_track_interpolates_gaps():
    lifter = skeleton(500)
    frames = [lifter] * 60 + [skeleton(1500, 400, 0.5)] * 10 + [lifter] * 60
    t = P.build_track(detections(frames))
    assert not t.found[60:70].any()
    assert np.allclose(t.image[65, P.L_HIP], lifter[P.L_HIP], atol=1)
    assert t.others == 10


def test_detection_cache_round_trip(tmp_path):
    det = detections([skeleton(500)] * 20)
    det.meta = {"v": 1, "size": 10}
    det.save(tmp_path / "d.npz")
    back = P.Detections.load(tmp_path / "d.npz")
    assert np.allclose(back.image, det.image) and (back.count == det.count).all()
    assert (back.fps, back.stride, back.width) == (det.fps, det.stride, det.width)
