"""Build the result page: a browser-playable copy of the video plus index.html with the data inlined.

The page draws the skeleton, bar path and angle labels itself, so they can be toggled and the
video scrubbed frame by frame. Opens straight from disk; no server needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from . import geometry
from . import pose as P
from .lifts.base import Lift, Signals
from .pose import PoseTrack

TEMPLATE = Path(__file__).with_name("viewer.html")


def _transcode(video: Path, dst: Path, fps: float) -> None:
    """H.264 at a constant frame rate so browser frame times line up with our frame indices."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required to build the viewer.")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-c:v", "libx264", "-preset", "fast",
         "-crf", "20", "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}", "-g", "5",
         "-movflags", "+faststart", str(dst)],
        check=True,
    )


def build(video: Path, out_dir: Path, t: PoseTrack, lift: Lift, sig: Signals, result: dict) -> Path:
    _transcode(video, out_dir / "video.mp4", t.fps)

    def r1(a) -> list:
        return [round(float(x), 1) for x in a]

    data = {
        "video": "video.mp4",
        "name": video.name,
        "lift": result["lift"],
        "fps": t.fps,
        "width": t.width,
        "height": t.height,
        "n": len(t),
        "skeleton": P.SKELETON,
        "lm": t.image.round().astype(int).tolist(),
        "vis": (t.vis > 0.3).astype(int).tolist(),
        "path": geometry.mid(t, lift.path_point).round().astype(int).tolist(),
        "pathName": lift.path,
        "series": [{"label": s.label, "unit": s.unit, "values": r1(sig[s.key]),
                    "tag": list(geometry.JOINTS[s.tag]) if s.tag else None} for s in lift.chart],
        "phases": result["phases"],
        "reps": result["reps"],
        "ignored": result["ignored"],
        "setup": result["setup"],
        "proportions": result["proportions"],
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = out_dir / "index.html"
    page.write_text(TEMPLATE.read_text().replace("/*DATA*/null", payload))
    return page
