"""Regression checks on real clips, from cached pose tracking.

Runs only for clips in tests/regression.toml whose tracking is cached in out/<clip>/detections.npz
(made by `scripts/regress.py --track-only` or any `form-boss analyze` run); the clips themselves are
not in git. For the full comparison against the saved baseline, run scripts/regress.py.
"""

import tomllib
from pathlib import Path

import pytest

from form_boss import pose
from form_boss.cli import analyze

ROOT = Path(__file__).resolve().parents[1]
CLIPS = tomllib.loads((ROOT / "tests" / "regression.toml").read_text())["clip"]


def cached(clip: dict) -> pose.PoseTrack:
    cache = ROOT / "out" / Path(clip["path"]).stem / "detections.npz"
    if not cache.exists():
        pytest.skip(f"no cached tracking for {clip['path']}")
    return pose.build_track(pose.Detections.load(cache))


@pytest.mark.parametrize("clip", [c for c in CLIPS if "reps" in c], ids=lambda c: Path(c["path"]).stem)
def test_rep_count(clip, tmp_path):
    res = analyze(ROOT / clip["path"], clip["lift"], tmp_path, page=False, track=cached(clip))
    assert len(res["reps"]) == clip["reps"]


@pytest.mark.parametrize("clip", CLIPS, ids=lambda c: Path(c["path"]).stem)
def test_lift_detected(clip, tmp_path):
    res = analyze(ROOT / clip["path"], None, tmp_path, page=False, track=cached(clip))
    assert res["lift"] == clip["lift"], res["lift_detected"]["reason"]
