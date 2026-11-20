"""Working-set state machine on synthetic rep lists."""

import numpy as np

from form_boss.reps import Rep
from form_boss.sets import find_sets, out_of_position, stepping

FPS = 30.0


def rep(start_s: float, dur_s: float = 2.0, partial: bool = False) -> Rep:
    s = int(start_s * FPS)
    e = s + int(dur_s * FPS)
    return Rep(start=s, bottom=(s + e) // 2, end=e, partial=partial)


def run(reps, n_s=120.0, posture=None):
    ignored = []
    if posture is not None:
        reps, ignored = out_of_position(reps, posture)
    kept, more, sets, phases = find_sets(reps, FPS, int(n_s * FPS))
    return kept, sorted(ignored + more, key=lambda i: i.rep.start), sets, phases


def test_one_set_keeps_every_rep():
    reps = [rep(t) for t in (10, 14, 18, 22, 26)]
    kept, ignored, sets, phases = run(reps)
    assert kept == reps and not ignored
    assert len(sets) == 1 and sets[0].reps == [0, 1, 2, 3, 4]
    assert [p.name for p in phases] == ["setup", "working set", "teardown"]


def test_long_pause_inside_set_is_not_a_new_set():
    # 11 s breathing pause between rep 2 and 3, as in the user squat clips.
    reps = [rep(t) for t in (84, 90, 103.7, 108.5, 115.2)]
    _, ignored, sets, _ = run(reps)
    assert len(sets) == 1 and not ignored


def test_lone_dips_before_and_after_the_set_are_setup_and_teardown():
    reps = [rep(5), *(rep(t) for t in (40, 44, 48, 52)), rep(90)]
    kept, ignored, sets, _ = run(reps)
    assert len(kept) == 4
    assert [i.kind for i in ignored] == ["setup", "teardown"]
    assert (sets[0].start, sets[0].end) == (reps[1].start, reps[4].end)


def test_two_sets_with_rest_between():
    reps = [rep(t) for t in (10, 14, 18, 60, 64, 68)]
    kept, ignored, sets, phases = run(reps)
    assert len(kept) == 6 and not ignored
    assert [s.reps for s in sets] == [[0, 1, 2], [3, 4, 5]]
    assert [p.name for p in phases] == ["setup", "working set", "rest", "working set", "teardown"]
    assert [p.set_number for p in phases if p.name == "working set"] == [1, 2]


def test_lone_movement_between_sets():
    reps = [rep(t) for t in (10, 14, 18, 40, 70, 74, 78)]
    kept, ignored, sets, _ = run(reps)
    assert [i.kind for i in ignored] == ["between"] and len(sets) == 2


def test_heavy_singles_are_all_kept():
    reps = [rep(t) for t in (10, 50, 90)]
    kept, ignored, sets, _ = run(reps)
    assert kept == reps and not ignored and len(sets) == 3


def test_hold_is_not_a_rep():
    # Standing with the bar for 10 s reads as one long "rep" of the rep signal.
    reps = [rep(5, dur_s=10), *(rep(t, dur_s=1.0) for t in (20, 22, 24, 26))]
    kept, ignored, *_ = run(reps)
    assert len(kept) == 4 and ignored[0].kind == "hold"


def test_partial_rep_is_never_a_hold():
    reps = [rep(0, dur_s=9, partial=True), *(rep(t, dur_s=1.0) for t in (12, 14, 16))]
    kept, ignored, *_ = run(reps)
    assert len(kept) == 4 and not ignored


def test_standing_up_between_rows_is_set_aside():
    # Torso hinged ~45 deg while rowing; upright (~5 deg) for the two "reps" while standing.
    n = int(120 * FPS)
    lean = np.full(n, 45.0)
    stand_reps = [rep(5), rep(60)]
    for r in stand_reps:
        lean[r.start:r.end + 1] = 5.0
    reps = sorted([*stand_reps, *(rep(t, 1.0) for t in (20, 22, 24, 26, 70, 72, 74))], key=lambda r: r.start)
    kept, ignored, sets, _ = run(reps, posture=lean)
    assert len(kept) == 7
    assert [i.kind for i in ignored] == ["posture", "posture"]
    assert len(sets) == 2


def test_rows_all_too_upright_are_kept_and_judged():
    n = int(60 * FPS)
    lean = np.full(n, 15.0)
    reps = [rep(t, 1.0) for t in (10, 12, 14, 16)]
    kept, ignored, *_ = run(reps, n_s=60, posture=lean)
    assert kept == reps and not ignored


def feet(n: int, walk: list[tuple[int, int]]) -> np.ndarray:
    """(n, 2, 2) ankle pixels: planted at x=400/480 with 2 px jitter, moving 150 px during `walk`."""
    rng = np.random.default_rng(0)
    a = np.zeros((n, 2, 2))
    a[:, 0, 0], a[:, 1, 0], a[..., 1] = 400, 480, 1000
    a += rng.normal(0, 2, a.shape)
    for s, e in walk:
        a[s:e + 1, :, 0] += np.linspace(0, 150, e - s + 1)[:, None]
        a[e + 1:, :, 0] += 150
    return a


def test_walking_dip_is_set_aside_and_planted_reps_kept():
    reps = [rep(t) for t in (10, 14, 18, 22)]
    n = int(40 * FPS)
    walk = [(reps[0].start, reps[0].end)]       # the first "rep" is a walk-out
    leg = np.full(n, 500.0)
    kept, ignored = stepping(reps, feet(n, walk), np.full((n, 2), 0.9), leg, FPS)
    assert kept == reps[1:] and [i.kind for i in ignored] == ["stepping"]


def test_hidden_feet_are_not_judged():
    reps = [rep(10)]
    n = int(20 * FPS)
    kept, ignored = stepping(reps, feet(n, [(reps[0].start, reps[0].end)]), np.full((n, 2), 0.2),
                             np.full(n, 500.0), FPS)
    assert kept == reps and not ignored
