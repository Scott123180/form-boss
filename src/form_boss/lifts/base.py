"""What a lift definition has to provide."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..errors import ConfigError
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


# How far a measurement may stray from the ideal before it is flagged, relative to the defaults.
STRICTNESS = {"lenient": 1.3, "normal": 1.0, "strict": 0.75}


@dataclass
class Profile:
    """User settings the checks take into account."""
    strictness: str = "normal"   # a key of STRICTNESS
    variation: str = ""          # a key of the lift's `variations`; "" = the lift's first one

    def tol(self, x: float) -> float:
        """Scale an allowed deviation (degrees, cm, % past the ideal) by the strictness."""
        return x * STRICTNESS[self.strictness]

    def limit(self, ideal: float, allowed: float) -> float:
        """A threshold `allowed` away from `ideal`, with the gap scaled by the strictness."""
        return ideal + self.tol(allowed - ideal)


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
    # Dips quicker than this are setup movements (unracking, stepping under the bar), not reps; see
    # reps.segment. Off by default: without a bar to pick up there is little to confuse with a rep,
    # and a fast bodyweight rep can genuinely take half a second.
    min_rep_s: float = 0.0
    min_phase_s: float = 0.0
    # Dips shallower than this share of the clip's typical rep are setup too (ducking under the bar,
    # walking it out). Off by default: it needs reps of a steady depth to compare against.
    min_depth_frac: float = 0.0
    joints: tuple[str, ...] = ("hip", "knee", "ankle")  # joints the checks depend on
    path: str = "bar"                                 # what the traced point is called
    path_point: str = "wrist"                         # joint whose left/right midpoint is traced
    camera: str = ""                                  # lift-specific camera placement advice
    # Named styles of the lift that change what good form looks like, e.g. low-bar vs high-bar
    # squat: {name: description}. The first is the default.
    variations: dict[str, str] = field(default_factory=dict)
    # Optional per-frame value that is high in the lift's working position (e.g. torso lean for a
    # row). Reps done well outside it are the lifter standing up between sets; see sets.find_sets.
    posture: Callable[[Signals, Geometry], np.ndarray] | None = None
    # Feet stay put through every rep, so a dip while a foot moves is walking, not a rep. Off where
    # the feet are often hidden (under a bench) and their landmarks are guesses.
    planted_feet: bool = True
    # Other views the lift has checks for, e.g. ("front",) when knee tracking is judged from the front.
    also_views: tuple[str, ...] = ()
    # Chart for a front or rear view, when that view measures different things; up to three.
    front_chart: list[Series] = field(default_factory=list)

    def charts(self, view: str) -> list[Series]:
        return self.front_chart if view == "front" and self.front_chart else self.chart

    def profile(self, strictness: str = "normal", variation: str = "") -> Profile:
        if strictness not in STRICTNESS:
            raise ConfigError(f"strictness must be one of {', '.join(STRICTNESS)}")
        if variation and variation not in self.variations:
            known = ", ".join(self.variations) or "none"
            raise ConfigError(f"{self.name} has no variation {variation!r} (known: {known})")
        return Profile(strictness, variation or next(iter(self.variations), ""))


def tempo_finding(m: Metrics, first: str, second: str, min_lowering_s: float) -> Finding:
    """Shared tempo check. `first`/`second` name the two halves of the rep; the half called
    "lowering" (or "descent") is the one that should be controlled."""
    a, b = m[first + "_s"], m[second + "_s"]
    msg = f"{first.capitalize()} {a:.1f}s, {second} {b:.1f}s."
    lowering = a if first in ("lowering", "descent") else b
    if lowering < min_lowering_s:
        return Finding("tempo", "warn", f"{msg} That is a fast drop; control the way down.")
    return Finding("tempo", "info", msg)
