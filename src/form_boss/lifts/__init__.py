"""Lift definitions. Add a lift by writing a module with a `LIFT` object and registering it here."""

from __future__ import annotations

from . import bent_over_row, bicep_curl, overhead_press, push_up, side_bend, squat, triceps_extension
from .base import Lift

LIFTS: dict[str, Lift] = {m.LIFT.name: m.LIFT for m in (
    squat, push_up, bicep_curl, overhead_press, triceps_extension, bent_over_row, side_bend)}
