"""Run the analysis over the labelled Mendeley clips and compare clips labelled good vs bad.

    uv run python scripts/eval_mendeley.py                  # every exercise that has been fetched
    uv run python scripts/eval_mendeley.py push_up squat    # just these
    uv run python scripts/eval_mendeley.py --pages          # also build the result page per clip

Writes out/mendeley_eval/<exercise>.csv (one row per clip) and prints a summary. Pose tracking is the
slow part and is cached in out/mendeley/.tracks; delete that folder after changing pose.py.
"""

import csv
import hashlib
import pickle
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from form_boss import pose
from form_boss.cli import analyze

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mendeley"
OUT = ROOT / "out" / "mendeley"
TRACKS = OUT / ".tracks"

# dataset folder -> (lift it actually shows, camera view that lift is judged from)
EXERCISES = {
    "squat": ("squat", "side"),
    "push_up": ("push_up", "side"),
    "bicep_curl": ("bicep_curl", "side"),
    "back": ("bent_over_row", "side"),
    "shoulder": ("overhead_press", "front"),
    "tricep": ("triceps_extension", "front"),
    "abs": ("side_bend", "front"),
}
PAGES = "--pages" in sys.argv


def run(job: tuple[Path, str]) -> dict:
    video, lift = job
    row = {"clip": video.stem, "label": video.parent.name}
    cache = TRACKS / (video.stem + ".pkl")
    try:
        if cache.exists():
            track = pickle.loads(cache.read_bytes())
        else:
            track = pose.track(video)
            cache.write_bytes(pickle.dumps(track))
        res = analyze(video, lift, OUT / video.stem, page=PAGES, track=track)
    except SystemExit as e:
        return {**row, "error": str(e)}
    reps = res["reps"]
    row |= {"view": res["view"], "setup_issues": sum(not s["ok"] for s in res["setup"]),
            "reps": len(reps), "skipped": res["reps_skipped"]}
    if reps:
        for k in reps[0]["metrics"]:
            row[k] = round(st.median(r["metrics"][k] for r in reps), 1)
        for chk in dict.fromkeys(f["check"] for r in reps for f in r["findings"]):
            found = [f for r in reps for f in r["findings"] if f["check"] == chk]
            if all(f["message"].startswith("Not assessed") for f in found):
                continue
            row["flag_" + chk.replace("/", "_").replace(" ", "_")] = round(
                sum(f["status"] in ("fault", "warn") for f in found) / len(reps), 2)
    return row


def usable(r: dict) -> bool:  # enough cleanly tracked reps to say anything
    return r.get("reps", 0) >= 3 and r["skipped"] <= r["reps"]


def evaluate(exercise: str, ex: ProcessPoolExecutor) -> None:
    lift, view = EXERCISES[exercise]
    # The dataset contains the same file under more than one name/label; keep only unambiguous clips.
    by_hash: dict[str, list[Path]] = {}
    for v in sorted((DATA / exercise / view).glob("*/*.mp4")):
        by_hash.setdefault(hashlib.md5(v.read_bytes()).hexdigest(), []).append(v)
    videos = sorted(vs[0] for vs in by_hash.values() if len(vs) == 1)
    dupes = sorted(v.stem for vs in by_hash.values() if len(vs) > 1 for v in vs)
    print(f"\n{'=' * 100}\n{exercise} -> {lift}, {view} view: {len(videos)} unique clips"
          + (f"; dropped {len(dupes)} duplicated files: {', '.join(dupes)}" if dupes else ""))
    if not videos:
        print(f"  nothing fetched; run: uv run python scripts/fetch_mendeley.py {exercise} --view {view}")
        return

    rows = list(ex.map(run, [(v, lift) for v in videos]))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    (ROOT / "out" / "mendeley_eval").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "out" / "mendeley_eval" / f"{exercise}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    cols = [k for k in keys if k not in ("clip", "label", "view", "error", "setup_issues", "reps", "skipped")]
    sel = {lab: [r for r in rows if r["label"] == lab and usable(r)] for lab in ("good", "bad")}
    for lab in sel:
        n = sum(r["label"] == lab for r in rows)
        wrong_view = sum(r.get("view") != view for r in sel[lab])
        print(f"  {lab}: {len(sel[lab])} of {n} clips tracked well enough to assess"
              + (f" ({wrong_view} of them not a true {view} view)" if wrong_view else ""))
    print(f"\n  {'median over clips':32} {'good':>8} {'bad':>8}")
    for k in cols:
        g, b = ([r[k] for r in sel[lab] if k in r] for lab in ("good", "bad"))
        if g and b:
            print(f"  {k:32} {st.median(g):8.1f} {st.median(b):8.1f}" if not k.startswith("flag_")
                  else f"  {k:32} {st.mean(g):8.2f} {st.mean(b):8.2f}   (mean share of reps flagged)")

    # Same person, good clip vs bad clip: does the bad clip get flagged more?
    subj: dict[str, dict] = {}
    for lab in sel:
        for r in sel[lab]:
            subj.setdefault(r["clip"][8:11], {})[lab] = r
    pairs = [d for d in subj.values() if len(d) == 2]
    print(f"\n  Paired by subject ({len(pairs)} people with both clips usable): which clip has more reps flagged")
    print(f"  {'check':32} {'bad':>5} {'good':>5} {'same':>5}")
    flags = [k for k in cols if k.startswith("flag_")]
    for k in flags + ["any check"]:
        def rate(r: dict) -> float:
            return max(r.get(f, 0) for f in flags) if k == "any check" else r.get(k, 0)
        d = [rate(p["bad"]) - rate(p["good"]) for p in pairs]
        more_bad, more_good = sum(x >= 0.2 for x in d), sum(x <= -0.2 for x in d)
        print(f"  {k.removeprefix('flag_'):32} {more_bad:5} {more_good:5} {len(d) - more_bad - more_good:5}")


def main() -> None:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(EXERCISES)
    TRACKS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        for name in names:
            evaluate(name, ex)


if __name__ == "__main__":
    main()
