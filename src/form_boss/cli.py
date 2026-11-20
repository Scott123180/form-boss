"""form-boss analyze <video> [--lift <name>]"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import geometry, overlay, pose, reps as reps_mod, sets as sets_mod, setup_check, viewer
from .errors import ConfigError, FormBossError
from .lifts import LIFTS
from .lifts.base import STRICTNESS, Finding, Lift, RepReport
from .lifts.detect import guess_lift

log = logging.getLogger("form_boss")

MARK = {"good": "ok  ", "warn": "WARN", "fault": "FAULT", "info": "    "}


def analyze(video: Path, lift_name: str | None, out_dir: Path, burn_in: bool = False,
            page: bool = True, track: pose.PoseTrack | None = None, options: dict | None = None,
            retrack: bool = False) -> dict:
    """Analyze one clip.

    lift_name: a key of LIFTS, or None/"auto" to work out the lift from the movement.
    options: {"strictness": ..., "variation": ...}; see Lift.profile.
    track: reuse pose tracking from an earlier run. Otherwise tracking is cached in
    out_dir/detections.npz and reused while the video file is unchanged, unless `retrack`.
    """
    options = options or {}
    if track is None:
        cache = out_dir / "detections.npz"
        if retrack and cache.exists():
            cache.unlink()
        track = pose.track(video, cache)
    # Measure on a copy corrected for camera tilt; draw and check framing on the original.
    mtrack, g = geometry.calibrate(track)
    detected = None
    if not lift_name or lift_name == "auto":
        detected = guess_lift(mtrack, g)
        lift_name = detected.lift
        log.info("Lift detected as %s (%s)", lift_name, detected.reason)
    if lift_name not in LIFTS:
        raise ConfigError(f"Unknown lift {lift_name!r}; choose from {', '.join(sorted(LIFTS))}")
    lift = LIFTS[lift_name]
    profile = lift.profile(options.get("strictness", "normal"), options.get("variation", ""))
    setup = setup_check.check(track, g, lift)
    sig = lift.signals(mtrack, g)
    reps, ignored = reps_mod.segment(sig[lift.rep_signal], track.fps, lift.min_dip, lift.near,
                                     lift.min_rep_s, lift.min_phase_s, lift.min_depth_frac)
    # Reps the analysis cannot see well enough: the lifter was lost, or walked so close to the
    # camera that the joints this lift is judged on left the picture (and were guessed).
    visible = geometry.in_frame(track, lift.joints)
    out_of_frame = [r for r in reps if visible[r.start:r.end + 1].mean() < 0.85]
    ignored += [reps_mod.Ignored(r, "frame") for r in out_of_frame]
    reps = [r for r in reps if r not in out_of_frame]
    tracked = [r for r in reps if track.found[r.start:r.end + 1].mean() >= 0.85]
    skipped = len(reps) - len(tracked)
    reps = tracked
    if skipped:
        setup.append(setup_check.SetupFinding(
            "skipped reps", False, f"{skipped} rep(s) left out because tracking was unreliable during them",
            "See the tracking advice above."))
    if lift.planted_feet:
        ankles = mtrack.image[:, list(geometry.JOINTS["ankle"])]
        leg = np.linalg.norm(geometry.mid(mtrack, "hip") - geometry.mid(mtrack, "ankle"), axis=-1)
        reps, walking = sets_mod.stepping(reps, ankles, track.vis[:, list(geometry.JOINTS["ankle"])], leg,
                                          track.fps)
        ignored += walking
    if lift.posture:
        reps, off = sets_mod.out_of_position(reps, lift.posture(sig, g))
        ignored += off
    reps, shallow = reps_mod.drop_shallow(reps, sig[lift.rep_signal], lift.min_depth_frac)
    reps, outside, work_sets, phases = sets_mod.find_sets(reps, track.fps, len(track))
    ignored = sorted(ignored + shallow + outside, key=lambda i: i.rep.start)

    reports: list[RepReport] = []
    for n, r in enumerate(reps, 1):
        m = lift.metrics(sig, g, r, track.fps)
        findings = []
        for chk in lift.checks:
            f: Finding = chk(m, g, profile)
            if f.check == "tempo" and r.partial:
                continue
            if f.needs_view and g.view != f.needs_view:
                f = Finding(f.check, "info", f"Not assessed: needs a {_view_name(f.needs_view)} view.",
                            f.needs_view)
            findings.append(f)
        reports.append(RepReport(n, r.start / track.fps, r.bottom / track.fps, r.end / track.fps,
                                 r.partial, m, findings))

    fps = track.fps
    rep_no = {id(r): n for n, r in enumerate(reps, 1)}
    result = {
        "video": str(video),
        "lift": lift.name,
        "lift_detected": asdict(detected) if detected else None,
        "variation": profile.variation,
        "strictness": profile.strictness,
        "fps": fps,
        "source_fps": track.source_fps or fps,
        "frame_stride": track.stride,
        "proportions": {
            "femur_cm": g.femur * 100, "tibia_cm": g.tibia * 100, "torso_cm": g.torso * 100,
            "femur_to_torso": g.femur / g.torso,
        },
        "view_yaw_deg": g.view_yaw,
        "view": g.view,
        "camera": {"pitch_deg": g.camera_pitch, "roll_deg": g.camera_roll, "corrected": g.corrected},
        "phases": list(lift.phases),
        "measurement_mode": g.mode,
        "reps_skipped": skipped,
        "ignored": [_ignored(i, lift, fps) for i in ignored],
        "sets": [{"number": n, "start_s": s.start / fps, "end_s": s.end / fps,
                  "reps": [rep_no[id(reps[i])] for i in s.reps]} for n, s in enumerate(work_sets, 1)],
        "timeline": [{"phase": p.name, "set": p.set_number, "start_s": p.start / fps,
                      "end_s": p.end / fps} for p in phases],
        "setup": [asdict(s) for s in setup],
        "reps": [asdict(r) for r in reports],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    if page:
        result["page"] = str(viewer.build(video, out_dir, track, lift, sig, result).resolve())
    if burn_in:
        result["overlay"] = str(overlay.render(video, out_dir / "overlay.mp4", track, lift, sig, reps,
                                                    reports, g.view))
    return result


def _is_view_skip(f: dict) -> bool:
    return f["status"] == "info" and f["message"].startswith("Not assessed: needs a")


def not_assessed(res: dict) -> dict[str, list[str]]:
    """Checks every rep skipped for want of another camera view: {view: [check, ...]}."""
    out: dict[str, list[str]] = {}
    if not res["reps"]:
        return out
    for f in res["reps"][0]["findings"]:
        if all(any(g["check"] == f["check"] and _is_view_skip(g) for g in r["findings"]) for r in res["reps"]):
            out.setdefault(f["needs_view"], []).append(f["check"])
    return out


def _view_name(v: str) -> str:
    return {"side": "true side", "front": "front or rear"}.get(v, v)


IGNORED_REASON = {
    "posture": "not in the working position: {pct:.0%} as far into it as your typical rep "
               "(standing up between sets?)",
    "hold": "too long for a rep: {value:.1f}s (holding or resting in place)",
    "setup": "a lone movement before the set started",
    "teardown": "a lone movement after the set ended",
    "between": "a lone movement between sets",
    "frame": "{joints} out of the picture for part of it",
    "stepping": "a foot moved to a new spot: walking, not a rep",
}


def _ignored(i: reps_mod.Ignored, lift: Lift, fps: float) -> dict:
    if i.kind == "shallow":
        reason = f"too shallow for a rep: {i.value:.0%} as deep as your typical rep"
    elif i.kind == "quick":
        part = "the whole movement" if i.part < 0 else f"the {lift.phases[i.part]}"
        reason = f"too quick for a rep: {part} took {i.value:.1f}s"
    else:
        joints = "/".join(lift.joints)
        reason = IGNORED_REASON[i.kind].format(value=i.value, pct=i.value, joints=joints)
    return {"start_s": i.rep.start / fps, "end_s": i.rep.end / fps, "kind": i.kind, "reason": reason}


def print_report(res: dict) -> None:
    det = res.get("lift_detected")
    how = f"  (detected: {det['reason']})" if det else ""
    var = f", {res['variation'].replace('_', '-')}" if res.get("variation") else ""
    print(f"\n{res['lift'].upper()}{var}  {res['video']}{how}")
    print("\nCamera setup")
    for s in res["setup"]:
        print(f"  [{'ok  ' if s['ok'] else 'FIX '}] {s['name']}: {s['detail']}")
        if s["fix"]:
            print(f"         -> {s['fix']}")
    p = res["proportions"]
    print(f"\nProportions (estimated): femur {p['femur_cm']:.0f} cm, torso {p['torso_cm']:.0f} cm, "
          f"femur/torso {p['femur_to_torso']:.2f}")
    if res["sets"]:
        print("\n" + "; ".join(f"Set {s['number']}: {s['start_s']:.1f}-{s['end_s']:.1f}s, "
                               f"{len(s['reps'])} rep{'s' * (len(s['reps']) != 1)}" for s in res["sets"]))
    if res["ignored"]:
        print(f"\nIgnored {len(res['ignored'])} movement(s) that are not working reps:")
        for i in res["ignored"]:
            print(f"  {i['start_s']:.1f}-{i['end_s']:.1f}s: {i['reason']}")
    if not res["reps"]:
        print("\nNo reps detected.")
    skipped = not_assessed(res)
    if skipped:
        print("\nNot assessed from this camera view: " + "; ".join(
            f"{', '.join(checks)} (need a {_view_name(v)} view)" for v, checks in skipped.items()))
    for r in res["reps"]:
        tag = " (partial: clip cuts into this rep)" if r["partial"] else ""
        print(f"\nRep {r['number']}  {res['phases'][1]} at {r['bottom_s']:.1f}s{tag}")
        for f in r["findings"]:
            if not _is_view_skip(f):
                print(f"  [{MARK[f['status']]:5}] {f['check']}: {f['message']}")
    if "page" in res:
        print(f"\nResult page: {res['page']}")
    if "overlay" in res:
        print(f"Burned-in video: {res['overlay']}")


def main(argv: list[str] | None = None) -> None:
    def verbosity(p: argparse.ArgumentParser, default) -> None:
        p.add_argument("-v", "--verbose", action="count", default=default,
                       help="more progress output (-vv for debug)")
        p.add_argument("-q", "--quiet", action="store_true", default=default or False,
                       help="only print the report and errors")

    ap = argparse.ArgumentParser(prog="form-boss")
    verbosity(ap, 0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze a lift video")
    verbosity(a, argparse.SUPPRESS)   # also accepted after the command
    a.add_argument("video", type=Path)
    a.add_argument("--lift", choices=["auto", *sorted(LIFTS)], default="auto",
                   help="the lift in the video (default: work it out from the movement)")
    a.add_argument("--variation", default="",
                   help="style of the lift, e.g. low_bar/high_bar for the squat; see `form-boss lifts`")
    a.add_argument("--strictness", choices=list(STRICTNESS), default="normal",
                   help="how far from ideal a rep may be before it is flagged")
    a.add_argument("--out", type=Path, help="output directory (default: out/<video name>)")
    a.add_argument("--burn-in", action="store_true", help="also write overlay.mp4 with drawings baked in")
    a.add_argument("--no-open", action="store_true", help="do not open the result page in a browser")
    a.add_argument("--retrack", action="store_true", help="ignore cached pose tracking for this video")
    sub.add_parser("lifts", help="list the lifts and their variations")
    args = ap.parse_args(argv)

    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose >= 2 else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    if args.cmd == "lifts":
        for name, lift in sorted(LIFTS.items()):
            also = "".join(f" (or the {v})" for v in lift.also_views)
            print(f"{name:18} film from the {lift.view}{also}")
            for v, desc in lift.variations.items():
                print(f"    --variation {v:14} {desc}")
        return

    out = args.out or Path("out") / args.video.stem
    try:
        res = analyze(args.video, args.lift, out, burn_in=args.burn_in, retrack=args.retrack,
                      options={"strictness": args.strictness, "variation": args.variation})
    except FormBossError as e:
        log.error("error: %s", e)
        sys.exit(1)
    print_report(res)
    if not args.no_open and "page" in res:
        webbrowser.open(Path(res["page"]).as_uri())


if __name__ == "__main__":
    main()
