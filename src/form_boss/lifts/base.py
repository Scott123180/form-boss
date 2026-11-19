"""What a lift definition has to provide."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..geometry import Geometry
from ..pose import PoseTrack
from ..reps import Rep

Signals = dict[str, np.ndarray]   # named per-frame series for one clip
Metrics = dict[str, float]        # named measurements for one rep


@dataclass
class Finding:
    check: str
    status: str   # "good" | "warn" | "fault" | "info"
    message: str
    needs_view: str = ""   # "side" or "front" when the check is only valid from that camera view


@dataclass
class RepReport:
    number: int
    start_s: float
    bottom_s: float   # the turnaround: deepest/peak position of the rep
    end_s: float
    partial: bool
    metrics: Metrics
    findings: list[Finding] = field(default_factory=list)


@dataclass
class Series:
    """A signal shown in the result page chart and as a live readout."""
    key: str
    label: str
    unit: str = "°"
    tag: str = ""   # joint to pin a live label to in the video: shoulder|elbow|wrist|hip|knee


@dataclass
class Lift:
    name: str
    view: str                                         # camera view to film from: "side" or "front"
    signals: Callable[[PoseTrack, Geometry], Signals]
    rep_signal: str                                   # key of the signal that dips once per rep
    metrics: Callable[[Signals, Geometry, Rep, float], Metrics]
    checks: list[Callable[[Metrics, Geometry], Finding]]
    chart: list[Series]                               # up to three
    phases: tuple[str, str, str] = ("descent", "bottom", "ascent")  # before / at / after the turnaround
    min_dip: float = 35.0                             # see reps.segment
    near: float = 12.0
    joints: tuple[str, ...] = ("hip", "knee", "ankle")  # joints the checks depend on
    path: str = "bar"                                 # what the traced point is called
    path_point: str = "wrist"                         # joint whose left/right midpoint is traced
    camera: str = ""                                  # lift-specific camera placement advice


def tempo_finding(m: Metrics, first: str, second: str, min_lowering_s: float) -> Finding:
    """Shared tempo check. `first`/`second` name the two halves of the rep; the half called
    "lowering" (or "descent") is the one that should be controlled."""
    a, b = m[first + "_s"], m[second + "_s"]
    msg = f"{first.capitalize()} {a:.1f}s, {second} {b:.1f}s."
    lowering = a if first in ("lowering", "descent") else b
    if lowering < min_lowering_s:
        return Finding("tempo", "warn", f"{msg} That is a fast drop; control the way down.")
    return Finding("tempo", "info", msg)
