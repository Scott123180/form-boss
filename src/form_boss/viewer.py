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
from .errors import DependencyError
from .lifts.base import Lift, Signals
from .pose import PoseTrack

TEMPLATE = Path(__file__).with_name("viewer.html")


# The page's copy of the video only has to show the lifter clearly behind the drawings: 1280 px on
# the long side and CRF 23 keep a two-minute 1080p phone clip around 35 MB instead of 200 MB.
MAX_SIDE = 1280
CRF = 23
GOP_S = 0.5   # a keyframe every half second keeps frame-by-frame scrubbing quick at 720p


def _transcode(video: Path, dst: Path, fps: float) -> None:
    """H.264 at the analysed frame rate, so browser frame times line up with our frame indices
    (frames the analysis skipped on high-fps footage are dropped here too)."""
    if not shutil.which("ffmpeg"):
        raise DependencyError("ffmpeg is required to build the result page.")
    scale = (f"scale='if(gte(iw,ih),min({MAX_SIDE},iw),-2)':'if(gte(iw,ih),-2,min({MAX_SIDE},ih))'"
             ":flags=bicubic")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-vf", scale, "-c:v", "libx264",
         "-preset", "fast", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}",
         "-g", str(max(1, round(fps * GOP_S))), "-movflags", "+faststart", str(dst)],
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
                    "tag": list(geometry.JOINTS[s.tag]) if s.tag else None}
                   for s in lift.charts(result["view"])],
        "phases": result["phases"],
        "reps": result["reps"],
        "ignored": result["ignored"],
        "sets": result["sets"],
        "timeline": result["timeline"],
        "variation": result["variation"],
        "strictness": result["strictness"],
        "detected": (result["lift_detected"] or {}).get("reason", ""),
        "setup": result["setup"],
        "proportions": result["proportions"],
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = out_dir / "index.html"
    page.write_text(TEMPLATE.read_text().replace("/*DATA*/null", payload))
    return page
