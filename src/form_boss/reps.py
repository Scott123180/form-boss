"""Rep segmentation from a signal that dips once per rep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class Rep:
    start: int    # frame the rep begins (signal leaves its resting level)
    bottom: int   # frame of the turnaround: deepest squat, top of a curl, lockout of a press
    end: int      # frame the lifter is back at the resting level
    partial: bool  # clip started or ended mid-rep, so tempo is unreliable


@dataclass
class Ignored:
    """A dip deep enough to count as a rep that is not one: unracking, racking, stepping in."""
    rep: Rep
    kind: str        # "quick": too fast for a rep; "shallow": far shallower than the other reps
    part: int = -1   # quick: what was too fast: -1 the whole movement, 0 the half before the turnaround, 2 after
    value: float = 0.0   # quick: seconds that part took; shallow: depth as a share of the typical rep's


def segment(signal: np.ndarray, fps: float, min_dip: float = 35.0, near: float = 12.0,
            min_rep_s: float = 0.0, min_phase_s: float = 0.0,
            min_depth_frac: float = 0.0) -> tuple[list[Rep], list[Ignored]]:
    """Find reps as dips in `signal` of at least `min_dip` below its resting level.

    The resting level is the top of the signal (knee angle while standing, for a squat). A rep runs
    from where the signal leaves that level (within `near` of it) to where it gets back. If the
    lifter turns around early, the rep ends at the highest point reached before the next one.

    Dips that are rep-deep but are not reps are returned as ignored instead:
    - quick: shorter than `min_rep_s` overall, or with either half shorter than `min_phase_s`.
      Setting the bar down, picking it up and stepping under it bend the joints as far as a rep
      but much faster. A half the clip cuts off is not timed.
    - shallow: less than `min_depth_frac` as deep as the typical rep (the 75th percentile of the
      dips left after the quick ones, so a few shallow ones do not drag it down). Ducking under the
      bar or walking it out is slow, but bends the joints about half as far as a working rep.
    Each ignored dip is removed before the remaining reps are re-measured, so it no longer splits a
    neighbouring rep in two.
    """
    top = float(np.percentile(signal, 95))
    near_top = top - near
    idx, _ = find_peaks(-signal, prominence=min_dip * 0.6, distance=max(1, int(fps * 0.7)))
    bottoms = [int(b) for b in idx if top - signal[b] >= min_dip]
    ignored: list[Ignored] = []
    while True:
        cands = _bound(signal, bottoms, near_top)
        worst = None   # the shortest offender goes first: removing it can lengthen its neighbours
        for i, (rep, cut_start, cut_end) in enumerate(cands):
            reason = _too_short(rep, cut_start, cut_end, fps, min_rep_s, min_phase_s)
            if reason is not None and (worst is None or rep.end - rep.start < worst[1].end - worst[1].start):
                worst = (i, rep, reason)
        if worst is None:
            break
        i, rep, reason = worst
        ignored.append(Ignored(rep, "quick", *reason))
        del bottoms[i]

    if min_depth_frac > 0 and bottoms:
        # Removing a shallow dip only lengthens its neighbours, so the quick check still holds.
        depth = np.array([top - signal[b] for b in bottoms])
        typical = float(np.percentile(depth, 75))
        shallow = depth < min_depth_frac * typical
        ignored += [Ignored(c[0], "shallow", value=float(d / typical))
                    for c, d, sh in zip(cands, depth, shallow) if sh]
        bottoms = [b for b, sh in zip(bottoms, shallow) if not sh]
        cands = _bound(signal, bottoms, near_top)
    return [c[0] for c in cands], sorted(ignored, key=lambda x: x.rep.start)


def _bound(signal: np.ndarray, bottoms: list[int], near_top: float) -> list[tuple[Rep, bool, bool]]:
    """Start and end of the rep around each bottom, and whether the clip cuts off its start/end."""
    out = []
    for n, b in enumerate(bottoms):
        lo = bottoms[n - 1] if n else 0
        hi = bottoms[n + 1] if n + 1 < len(bottoms) else len(signal) - 1
        s = b
        while s > lo and signal[s] < near_top:
            s -= 1
        if signal[s] < near_top and n:               # never got back up: split at the high point
            s = lo + int(np.argmax(signal[lo:b + 1]))
        e = b
        while e < hi and signal[e] < near_top:
            e += 1
        if signal[e] < near_top and n + 1 < len(bottoms):
            e = b + int(np.argmax(signal[b:hi + 1]))
        cut_start = bool(n == 0 and signal[s] < near_top)
        cut_end = bool(n + 1 == len(bottoms) and signal[e] < near_top)
        out.append((Rep(start=int(s), bottom=b, end=int(e), partial=cut_start or cut_end), cut_start, cut_end))
    return out


def _too_short(r: Rep, cut_start: bool, cut_end: bool, fps: float,
               min_rep_s: float, min_phase_s: float) -> tuple[int, float] | None:
    first, second, total = (r.bottom - r.start) / fps, (r.end - r.bottom) / fps, (r.end - r.start) / fps
    if not (cut_start or cut_end) and total < min_rep_s:
        return -1, total
    if not cut_start and first < min_phase_s:
        return 0, first
    if not cut_end and second < min_phase_s:
        return 2, second
    return None
