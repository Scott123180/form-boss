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


def segment(signal: np.ndarray, fps: float, min_dip: float = 35.0, near: float = 12.0) -> list[Rep]:
    """Find reps as dips in `signal` of at least `min_dip` below its resting level.

    The resting level is the top of the signal (knee angle while standing, for a squat). A rep runs
    from where the signal leaves that level (within `near` of it) to where it gets back. If the
    lifter turns around early, the rep ends at the highest point reached before the next one.
    """
    top = float(np.percentile(signal, 95))
    near_top = top - near
    idx, _ = find_peaks(-signal, prominence=min_dip * 0.6, distance=max(1, int(fps * 0.7)))
    bottoms = [int(b) for b in idx if top - signal[b] >= min_dip]
    reps = []
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
        partial = bool((n == 0 and signal[s] < near_top) or (n + 1 == len(bottoms) and signal[e] < near_top))
        reps.append(Rep(start=int(s), bottom=b, end=int(e), partial=partial))
    return reps
