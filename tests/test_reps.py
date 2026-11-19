"""Rep segmentation on synthetic joint-angle signals."""

import numpy as np

from form_boss.reps import segment

FPS = 30.0
STAND = 175.0


def dip(down_s: float, up_s: float, depth: float = 90.0) -> np.ndarray:
    """One rep: cosine ease from standing down `depth` degrees and back up."""
    d = STAND - depth / 2 * (1 - np.cos(np.linspace(0, np.pi, int(down_s * FPS), endpoint=False)))
    u = STAND - depth / 2 * (1 + np.cos(np.linspace(0, np.pi, int(up_s * FPS))))
    return np.concatenate([d, u])


def stand(s: float) -> np.ndarray:
    return np.full(int(s * FPS), STAND)


def clip(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


def times(reps) -> list[float]:
    return [round(r.bottom / FPS, 1) for r in reps]


def test_counts_reps_without_filter():
    sig = clip(stand(2), dip(1.2, 1.0), stand(1), dip(1.2, 1.0), stand(2))
    reps, ignored = segment(sig, FPS)
    assert len(reps) == 2 and not ignored
    assert not any(r.partial for r in reps)


def test_filter_off_by_default_keeps_quick_dips():
    sig = clip(stand(2), dip(0.2, 0.2, 60), stand(2), dip(1.2, 1.0), stand(2))
    reps, ignored = segment(sig, FPS)
    assert len(reps) == 2 and not ignored


def test_drops_quick_setup_dips_around_the_set():
    # unrack bob, three real reps, rack bob
    sig = clip(stand(1), dip(0.2, 0.2, 60), stand(3),
               dip(1.4, 1.0), stand(1), dip(1.4, 1.0), stand(1), dip(1.4, 1.0),
               stand(3), dip(0.15, 0.25, 50), stand(1))
    reps, ignored = segment(sig, FPS, min_rep_s=0.6, min_phase_s=0.25)
    assert len(reps) == 3
    assert len(ignored) == 2
    assert ignored[0].rep.start < reps[0].start and ignored[1].rep.start > reps[-1].end
    assert all(i.kind == "quick" and i.value < 0.6 for i in ignored)


def test_drops_fast_drop_slow_rise_by_phase():
    # long overall, but the joint snaps down in 0.15 s: ducking under the bar, not a squat
    sig = clip(stand(2), dip(0.15, 1.5, 70), stand(2), dip(1.2, 1.0), stand(2))
    reps, ignored = segment(sig, FPS, min_rep_s=0.6, min_phase_s=0.25)
    assert len(reps) == 1
    assert len(ignored) == 1 and ignored[0].part == 0


def test_keeps_fast_but_real_reps():
    # a quick bodyweight set, as fast as the quickest real reps in the Mendeley clips: timed from
    # where the angle leaves the resting band, about 0.3 s down and 0.35 s up
    sig = clip(stand(1), *[np.concatenate([dip(0.4, 0.45), stand(0.4)]) for _ in range(5)], stand(1))
    reps, ignored = segment(sig, FPS, min_rep_s=0.6, min_phase_s=0.25)
    assert len(reps) == 5 and not ignored


def test_ghost_between_reps_no_longer_splits_a_rep():
    # the lifter jolts down 40 deg mid-ascent without getting back up first: without the filter the
    # jolt is counted as a rep and steals the end of the real rep before it
    sig = clip(stand(2), dip(1.2, 1.5), stand(2))
    i = int((2 + 1.2 + 0.9) * FPS)   # 0.9 s into the ascent, still well below standing
    sig[i:i + 6] -= 40 * np.sin(np.linspace(0, np.pi, 6))
    unfiltered, _ = segment(sig, FPS, min_dip=20)
    reps, ignored = segment(sig, FPS, min_dip=20, min_rep_s=0.6, min_phase_s=0.25)
    assert len(unfiltered) == 2 and unfiltered[0].end < i + 6
    assert len(reps) == 1 and len(ignored) == 1
    assert reps[0].end > i + 6   # runs on to the real top, past the jolt


def test_clip_cut_half_is_not_timed():
    # clip starts 0.2 s before the bottom of a rep: the cut-off descent must not get it thrown out
    sig = clip(dip(0.6, 1.0, 120)[int(0.4 * FPS):], stand(1.5), dip(1.2, 1.0), stand(1.5))
    reps, ignored = segment(sig, FPS, min_rep_s=0.6, min_phase_s=0.25)
    assert len(reps) == 2 and reps[0].partial and not ignored


def test_drops_slow_shallow_setup_dips():
    # ducking under the bar (slow, half depth), five working reps, ducking out again
    work = [np.concatenate([dip(1.4, 1.1, 90), stand(1)]) for _ in range(5)]
    sig = clip(stand(1), dip(0.8, 0.8, 45), stand(3), *work, stand(2), dip(0.7, 0.9, 40), stand(1))
    reps, ignored = segment(sig, FPS, min_rep_s=0.6, min_phase_s=0.25, min_depth_frac=0.6)
    assert len(reps) == 5
    assert [i.kind for i in ignored] == ["shallow", "shallow"]
    assert all(0.4 < i.value < 0.55 for i in ignored)


def test_shallow_filter_off_by_default():
    sig = clip(stand(1), dip(0.8, 0.8, 45), stand(2), dip(1.4, 1.1, 90), stand(2))
    reps, ignored = segment(sig, FPS)
    assert len(reps) == 2 and not ignored


def test_keeps_reps_of_somewhat_varying_depth():
    # depth drifting from 95 to 65 deg over a set (fatigue) is still a set of reps
    work = [np.concatenate([dip(1.3, 1.0, d), stand(1)]) for d in (95, 90, 85, 75, 65)]
    reps, ignored = segment(clip(stand(1), *work, stand(1)), FPS, min_depth_frac=0.6)
    assert len(reps) == 5 and not ignored


def test_shallow_dip_inside_a_rep_is_merged_back():
    # a half-depth bounce on the way up that never returns to standing is part of the rep, not one
    up = dip(1.2, 1.4)
    sig = clip(stand(2), up, stand(2), dip(1.2, 1.2), stand(2))
    i = int((2 + 1.2 + 0.95) * FPS)
    sig[i:i + 20] -= 45 * np.sin(np.linspace(0, np.pi, 20))
    unfiltered, _ = segment(sig, FPS, min_dip=30)
    reps, ignored = segment(sig, FPS, min_dip=30, min_depth_frac=0.6)
    assert len(unfiltered) == 3
    assert len(reps) == 2 and len(ignored) == 1 and ignored[0].kind == "shallow"
    assert reps[0].end > i + 10   # runs past the bounce to the real top
