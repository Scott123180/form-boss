"""Working-set state machine: SETUP -> WORKING SET -> (REST -> WORKING SET ...) -> TEARDOWN.

Rep segmentation finds every dip that looks like a rep. Some of them happen outside the set: a lone
dip while walking the bar out, standing up with the bar between sets of rows, re-racking. This
step groups the reps into sets by rhythm and sets aside what falls outside them, so form findings
only come from reps inside a working set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reps import Ignored, Rep

# Reps further apart than this belong to different sets. Lifters breathe for several seconds
# between heavy reps, and the user squat clips have a 11 s pause inside one set, so this is generous.
MIN_SET_GAP_S = 15.0
SET_GAP_REPS = 5.0          # ... or this many typical rep durations, whichever is longer
# A "rep" this many times longer than the typical one is the lifter holding or resting in place.
MAX_REP_LENGTH_X = 3.0
MAX_REP_S = 12.0            # never a rep, whatever the others take: standing between sets


# A foot that ends a "rep" this far (share of leg length) from where it started, and wandered at
# least MAX_FOOT_SPAN on the way, was walking: stepping under the bar, walking it out, carrying it
# back. On the user clips planted feet score under a fifth of these limits, walking 1-5 times them.
# Ankle landmarks jitter a lot on low-resolution footage, so both have to be exceeded.
MAX_FOOT_NET = 0.15
MAX_FOOT_SPAN = 0.12
MIN_FOOT_VIS = 0.5   # a hidden ankle's landmark is a guess that wanders by half a leg length


def stepping(reps: list[Rep], ankles: np.ndarray, vis: np.ndarray, leg: np.ndarray, fps: float
             ) -> tuple[list[Rep], list[Ignored]]:
    """Set aside reps during which a visible foot moves to a new spot. `ankles` (T, 2, 2) left/right
    ankle pixels, `vis` (T, 2) their visibility, `leg` (T,) hip-to-ankle length in pixels."""
    w = max(2, int(0.2 * fps))
    kept, ignored = [], []
    for r in reps:
        sl = slice(r.start, r.end + 1)
        scale = float(np.median(leg[sl])) + 1e-9
        score = 0.0
        for side in (0, 1):
            if vis[sl, side].mean() < MIN_FOOT_VIS:
                continue
            a = ankles[:, side]
            net = np.linalg.norm(np.median(a[max(r.start, r.end - w + 1):r.end + 1], axis=0)
                                 - np.median(a[r.start:r.start + w], axis=0)) / scale
            span = np.linalg.norm(np.percentile(a[sl], 95, axis=0) - np.percentile(a[sl], 5, axis=0)) / scale
            score = max(score, min(net / MAX_FOOT_NET, span / MAX_FOOT_SPAN))
        if score > 1:
            ignored.append(Ignored(r, "stepping"))
        else:
            kept.append(r)
    return kept, ignored


@dataclass
class WorkingSet:
    start: int        # first frame of its first rep
    end: int          # last frame of its last rep
    reps: list[int]   # indices into the kept reps


@dataclass
class Phase:
    name: str         # "setup" | "working set" | "rest" | "teardown"
    start: int
    end: int
    set_number: int = 0   # 1-based for "working set"


def out_of_position(reps: list[Rep], posture: np.ndarray, frac: float = 0.5
                    ) -> tuple[list[Rep], list[Ignored]]:
    """Set aside reps done well outside the lift's working position.

    `posture` is a per-frame value that is high in the working position (torso lean for a row,
    hands up on the bar for a squat). A rep whose median is below `frac` of the typical rep's (75th
    percentile) is the lifter out of position: standing up between sets of rows, squatting down
    without the bar. If all the reps are low they stay in, and the checks judge them.
    """
    if len(reps) < 2:
        return reps, []
    val = np.array([np.median(posture[r.start:r.end + 1]) for r in reps])
    typical = float(np.percentile(val, 75))
    if typical <= 0:
        return reps, []
    low = val < frac * typical
    return ([r for r, lo in zip(reps, low) if not lo],
            [Ignored(r, "posture", value=float(v / typical)) for r, v, lo in zip(reps, val, low) if lo])


def find_sets(reps: list[Rep], fps: float, n_frames: int
              ) -> tuple[list[Rep], list[Ignored], list[WorkingSet], list[Phase]]:
    """Group `reps` into working sets and set aside what falls outside them.

    - Holds: with 3+ reps, a rep lasting over MAX_REP_LENGTH_X times the typical one; any rep over
      MAX_REP_S.
    - A lone rep more than a set gap away from any other is setup (before the first set of 2+
      reps), teardown (after the last) or a stray movement between sets. With no set of 2+ reps
      (heavy singles), nothing is dropped this way and each rep is its own set.
    """
    ignored: list[Ignored] = []
    kept = list(reps)

    dur = np.array([(r.end - r.start) / fps for r in kept])
    if len(kept):
        limit = MAX_REP_S
        if len(kept) >= 3:
            limit = min(limit, MAX_REP_LENGTH_X * float(np.median(dur)))
        hold = dur > limit
        ignored += [Ignored(r, "hold", value=float(d)) for r, d, h in zip(kept, dur, hold) if h and not r.partial]
        kept = [r for r, d, h in zip(kept, dur, hold) if not (h and not r.partial)]

    groups: list[list[int]] = []
    if kept:
        typical = float(np.median([(r.end - r.start) / fps for r in kept]))
        gap = max(MIN_SET_GAP_S, SET_GAP_REPS * typical) * fps
        groups = [[0]]
        for i in range(1, len(kept)):
            if kept[i].start - kept[i - 1].end > gap:
                groups.append([])
            groups[-1].append(i)
    real = [g for g in groups if len(g) >= 2]
    if real:
        lone = [g[0] for g in groups if len(g) < 2]
        for i in lone:
            where = ("setup" if i < real[0][0] else "teardown" if i > real[-1][-1] else "between")
            ignored.append(Ignored(kept[i], where))
        keep_idx = [i for i in range(len(kept)) if i not in lone]
        remap = {old: new for new, old in enumerate(keep_idx)}
        kept = [kept[i] for i in keep_idx]
        groups = [[remap[i] for i in g] for g in real]

    sets = [WorkingSet(kept[g[0]].start, kept[g[-1]].end, g) for g in groups]
    return kept, sorted(ignored, key=lambda x: x.rep.start), sets, timeline(sets, n_frames)


def timeline(sets: list[WorkingSet], n_frames: int) -> list[Phase]:
    if not sets:
        return []
    out: list[Phase] = []
    if sets[0].start > 0:
        out.append(Phase("setup", 0, sets[0].start - 1))
    for n, s in enumerate(sets, 1):
        if n > 1:
            out.append(Phase("rest", sets[n - 2].end + 1, s.start - 1))
        out.append(Phase("working set", s.start, s.end, n))
    if sets[-1].end < n_frames - 1:
        out.append(Phase("teardown", sets[-1].end + 1, n_frames - 1))
    return out
