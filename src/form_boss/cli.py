"""form-boss analyze <video> --lift <name>"""

from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import asdict
from pathlib import Path

from . import geometry, overlay, pose, reps as reps_mod, setup_check, viewer
from .lifts import LIFTS
from .lifts.base import Finding, RepReport

MARK = {"good": "ok  ", "warn": "WARN", "fault": "FAULT", "info": "    "}


def analyze(video: Path, lift_name: str, out_dir: Path, burn_in: bool = False,
            page: bool = True, track: pose.PoseTrack | None = None) -> dict:
    """Analyze one clip. Pass `track` to reuse pose tracking from an earlier run."""
    lift = LIFTS[lift_name]
    track = track or pose.track(video)
    g = geometry.compute(track)
    setup = setup_check.check(track, g, lift)
    sig = lift.signals(track, g)
    reps = reps_mod.segment(sig[lift.rep_signal], track.fps, lift.min_dip, lift.near)
    tracked = [r for r in reps if track.found[r.start:r.end + 1].mean() >= 0.85]
    skipped = len(reps) - len(tracked)
    reps = tracked
    if skipped:
        setup.append(setup_check.SetupFinding(
            "skipped reps", False, f"{skipped} rep(s) left out because tracking was unreliable during them",
            "See the tracking advice above."))

    reports: list[RepReport] = []
    for n, r in enumerate(reps, 1):
        m = lift.metrics(sig, g, r, track.fps)
        findings = []
        for chk in lift.checks:
            f: Finding = chk(m, g)
            if f.check == "tempo" and r.partial:
                continue
            if f.needs_view and g.view != f.needs_view:
                f = Finding(f.check, "info", f"Not assessed: needs a true {f.needs_view} view.", f.needs_view)
            findings.append(f)
        reports.append(RepReport(n, r.start / track.fps, r.bottom / track.fps, r.end / track.fps,
                                 r.partial, m, findings))

    result = {
        "video": str(video),
        "lift": lift.name,
        "fps": track.fps,
        "proportions": {
            "femur_cm": g.femur * 100, "tibia_cm": g.tibia * 100, "torso_cm": g.torso * 100,
            "femur_to_torso": g.femur / g.torso,
        },
        "view_yaw_deg": g.view_yaw,
        "view": g.view,
        "phases": list(lift.phases),
        "measurement_mode": g.mode,
        "reps_skipped": skipped,
        "setup": [asdict(s) for s in setup],
        "reps": [asdict(r) for r in reports],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    if page:
        result["page"] = str(viewer.build(video, out_dir, track, lift, sig, result).resolve())
    if burn_in:
        result["overlay"] = str(overlay.render(video, out_dir / "overlay.mp4", track, lift, sig, reps, reports))
    return result


def print_report(res: dict) -> None:
    print(f"\n{res['lift'].upper()}  {res['video']}")
    print("\nCamera setup")
    for s in res["setup"]:
        print(f"  [{'ok  ' if s['ok'] else 'FIX '}] {s['name']}: {s['detail']}")
        if s["fix"]:
            print(f"         -> {s['fix']}")
    p = res["proportions"]
    print(f"\nProportions (estimated): femur {p['femur_cm']:.0f} cm, torso {p['torso_cm']:.0f} cm, "
          f"femur/torso {p['femur_to_torso']:.2f}")
    if not res["reps"]:
        print("\nNo reps detected.")
    for r in res["reps"]:
        tag = " (partial: clip cuts into this rep)" if r["partial"] else ""
        print(f"\nRep {r['number']}  {res['phases'][1]} at {r['bottom_s']:.1f}s{tag}")
        for f in r["findings"]:
            print(f"  [{MARK[f['status']]:5}] {f['check']}: {f['message']}")
    if "page" in res:
        print(f"\nResult page: {res['page']}")
    if "overlay" in res:
        print(f"Burned-in video: {res['overlay']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="form-boss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze a lift video")
    a.add_argument("video", type=Path)
    a.add_argument("--lift", choices=sorted(LIFTS), default="squat")
    a.add_argument("--out", type=Path, help="output directory (default: out/<video name>)")
    a.add_argument("--burn-in", action="store_true", help="also write overlay.mp4 with drawings baked in")
    a.add_argument("--no-open", action="store_true", help="do not open the result page in a browser")
    args = ap.parse_args()
    out = args.out or Path("out") / args.video.stem
    res = analyze(args.video, args.lift, out, burn_in=args.burn_in)
    print_report(res)
    if not args.no_open:
        webbrowser.open(Path(res["page"]).as_uri())


if __name__ == "__main__":
    main()
