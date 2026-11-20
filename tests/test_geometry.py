"""Angle math, in-frame checks and user presets."""

import numpy as np
import pytest

from form_boss import geometry as G
from form_boss.errors import ConfigError
from form_boss.lifts import LIFTS
from form_boss.lifts.base import Profile
from form_boss.pose import PoseTrack


def test_angle_right_and_straight():
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0]])
    c = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert np.allclose(G._angle(a, b, c), [90.0, 180.0], atol=0.01)


def test_angle_3d_and_degenerate():
    a, b, c = np.array([[1.0, 0, 0]]), np.zeros((1, 3)), np.array([[0, 0, 2.0]])
    assert np.allclose(G._angle(a, b, c), 90.0)
    assert np.isfinite(G._angle(b, b, c)).all()   # zero-length segment does not blow up


def track(image: np.ndarray, w: int = 1000, h: int = 1000) -> PoseTrack:
    T = len(image)
    return PoseTrack(image=image, world=np.zeros((T, 33, 3)), vis=np.ones((T, 33)),
                     found=np.ones(T, bool), fps=30.0, width=w, height=h)


def test_tilt_sign():
    im = np.zeros((1, 33, 2))
    im[0, G.JOINTS["hip"][0]] = (500, 500)
    im[0, G.JOINTS["shoulder"][0]] = (600, 400)   # shoulder up and to the right
    assert np.allclose(G.tilt(track(im), "hip", "shoulder", 0), 45.0)


def test_in_frame():
    im = np.full((3, 33, 2), 500.0)
    im[1, G.JOINTS["ankle"][0]] = (500, 1200)          # one ankle below the picture: the other is fine
    im[2, list(G.JOINTS["ankle"])] = (500, 1200)       # both out
    assert G.in_frame(track(im), ("hip", "knee", "ankle")).tolist() == [True, True, False]


def test_relative_to_standing_follows_drift():
    # The lifter walks away (shrinks to 60%) and squats; reps read the same either side of the walk.
    fps = 30.0
    size = np.r_[np.full(300, 1.0), np.linspace(1.0, 0.6, 60), np.full(300, 0.6)]
    x = size.copy()
    for start in (100, 500):
        x[start:start + 30] *= 0.5
    rel = G.relative_to_standing(x, fps)
    assert abs(rel[115] - 0.5) < 0.05 and abs(rel[515] - 0.5) < 0.05
    assert abs(rel[200] - 1.0) < 0.05 and abs(rel[600] - 1.0) < 0.05


def test_profile_scaling():
    assert Profile("strict").tol(10) == 7.5 and Profile("lenient").tol(10) == 13.0
    assert Profile("strict").limit(180, 160) == 165.0
    assert Profile().limit(180, 160) == 160.0


def test_lift_profile_validates():
    sq = LIFTS["squat"]
    assert sq.profile().variation == "high_bar"
    assert sq.profile("strict", "low_bar").variation == "low_bar"
    with pytest.raises(ConfigError):
        sq.profile(variation="sumo")
    with pytest.raises(ConfigError):
        sq.profile(strictness="brutal")


def test_strictness_changes_a_verdict():
    from form_boss.lifts.squat import depth
    g = G.Geometry(*[np.zeros(1)] * 6, np.zeros((1, 2)), np.zeros((1, 2)), 0.45, 0.4, 0.5, 0.25, 90.0)
    m = {"thigh_angle_bottom": 4.5, "hip_height_bottom_pct": 50.0}
    assert depth(m, g, Profile("normal")).status == "good"
    assert depth(m, g, Profile("strict")).status == "warn"
