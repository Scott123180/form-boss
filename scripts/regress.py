"""Regression runner: analyse the clips in tests/regression.toml and compare with the saved baseline.

    uv run python scripts/regress.py               # analyse, compare with the baseline, print diffs
    uv run python scripts/regress.py --update      # ... and save the results as the new baseline
    uv run python scripts/regress.py --track-only  # only run (and cache) pose tracking
    uv run python scripts/regress.py --pages       # also build each clip's result page
    uv run python scripts/regress.py squat_side    # only clips whose path contains this

Pose tracking is cached next to each clip's results in out/<clip name>/detections.npz, the same
place `form-boss analyze` keeps it, so after the first run this takes seconds. Exits non-zero if a
clip's rep count differs from the count in the manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from form_boss import pose
from form_boss.cli import analyze
from form_boss.errors import FormBossError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "regression.toml"
BASELINE = ROOT / "tests" / "regression_baseline.json"


def out_dir(clip: dict) -> Path:
    return ROOT / "out" / Path(clip["path"]).stem


def summarize(res: dict) -> dict:
    """The parts of a result that should only change on purpose."""
    return {
        "lift": res["lift"],
        "view": res["view"],
        "reps": len(res["reps"]),
        "ignored": [f"{i['start_s']:.1f}s {i['kind']}" for i in res["ignored"]],
        "sets": [f"{s['start_s']:.1f}-{s['end_s']:.1f}s" for s in res.get("sets", [])],
        "findings": [{f["check"]: f["status"] for f in r["findings"]} for r in res["reps"]],
        "setup_fixes": sorted(s["name"] for s in res["setup"] if not s["ok"]),
    }


def track_one(clip: dict) -> str:
    video = ROOT / clip["path"]
    try:
        pose.detections(video, out_dir(clip) / "detections.npz")
    except FormBossError as e:
        return f"{clip['path']}: {e}"
    return f"{clip['path']}: tracked"


def run_one(job: tuple[dict, bool]) -> tuple[str, dict | str]:
    clip, pages = job
    video = ROOT / clip["path"]
    try:
        res = analyze(video, clip["lift"], out_dir(clip), page=pages,
                      options=clip.get("options", {}))
    except FormBossError as e:
        return clip["path"], str(e)
    return clip["path"], summarize(res)


def diff(old: dict, new: dict) -> list[str]:
    out = []
    for k in ("lift", "view", "reps", "ignored", "sets", "setup_fixes"):
        if old.get(k) != new.get(k):
            out.append(f"{k}: {old.get(k)} -> {new.get(k)}")
    of, nf = old.get("findings", []), new.get("findings", [])
    if len(of) == len(nf):
        for n, (a, b) in enumerate(zip(of, nf), 1):
            for chk in dict.fromkeys([*a, *b]):
                if a.get(chk) != b.get(chk):
                    out.append(f"rep {n} {chk}: {a.get(chk)} -> {b.get(chk)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="*")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--track-only", action="store_true")
    ap.add_argument("--pages", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    clips = tomllib.loads(MANIFEST.read_text())["clip"]
    clips = [c for c in clips if not args.filter or any(f in c["path"] for f in args.filter)]
    missing = [c["path"] for c in clips if not (ROOT / c["path"]).exists()]
    clips = [c for c in clips if (ROOT / c["path"]).exists()]
    if missing:
        print(f"Skipping {len(missing)} clip(s) not on this machine: {', '.join(missing)}")

    with ProcessPoolExecutor(args.jobs) as ex:
        for msg in ex.map(track_one, clips):
            print(msg)
        if args.track_only:
            return
        results = dict(ex.map(run_one, [(c, args.pages) for c in clips]))

    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    bad = 0
    for c in clips:
        p, res = c["path"], results[c["path"]]
        if isinstance(res, str):
            print(f"\n{p}\n  ERROR {res}")
            bad += 1
            continue
        want = c.get("reps")
        flag = "" if want is None or want == res["reps"] else f"   <-- expected {want}"
        bad += bool(flag)
        print(f"\n{p}  [{res['lift']}, {res['view']} view] {res['reps']} reps, "
              f"{len(res['ignored'])} ignored{flag}")
        if p in base:
            for line in diff(base[p], res):
                print(f"  changed  {line}")
        else:
            print("  (no baseline)")
    if args.update:
        base.update({p: r for p, r in results.items() if not isinstance(r, str)})
        BASELINE.write_text(json.dumps(base, indent=1) + "\n")
        print(f"\nBaseline saved to {BASELINE.relative_to(ROOT)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
