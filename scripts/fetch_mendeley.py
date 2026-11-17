"""Fetch clips from the Mendeley multi-view fitness dataset (CC BY 4.0).

  "A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels"
  https://data.mendeley.com/datasets/kgbb3yn47p  (doi:10.17632/kgbb3yn47p.3)

The dataset is a 3 GB zip; this pulls only the wanted clips using HTTP range requests.
About 25 subjects per exercise, each with a clip labelled good and one labelled bad, filmed from
the side, front and diagonal. Clips land in data/mendeley/<exercise>/<view>/<good|bad>/.

    uv run python scripts/fetch_mendeley.py squat                 # side view, good + bad
    uv run python scripts/fetch_mendeley.py push_up bicep_curl --view front
    uv run python scripts/fetch_mendeley.py all --subjects 3      # first 3 subjects of everything
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://data.mendeley.com/public-files/datasets/kgbb3yn47p/files/"
       "2499db21-0988-42a4-b068-0e79a8a3c51a/file_downloaded")
UA = {"User-Agent": "form-analysis-script/0.1"}
DEST = Path(__file__).resolve().parents[1] / "data" / "mendeley"
EXERCISES = ["abs", "back", "bicep_curl", "push_up", "shoulder", "squat", "tricep"]


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str):
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exercise", nargs="+", choices=EXERCISES + ["all"])
    ap.add_argument("--view", choices=["side", "front", "diagonal"], default="side")
    ap.add_argument("--subjects", type=int, help="only the first N subjects (by file order)")
    args = ap.parse_args()
    want = set(EXERCISES if "all" in args.exercise else args.exercise)
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 20))
    taken: dict[tuple[str, str], int] = {}
    for info in z.infolist():
        parts = info.filename.split("/")  # .../<exercise>/<good|bad>/<view>/subject_NNN_....mp4
        if info.is_dir() or len(parts) < 4:
            continue
        exercise, label, view = parts[-4], parts[-3], parts[-2]
        if exercise not in want or view != args.view:
            continue
        taken[exercise, label] = taken.get((exercise, label), 0) + 1
        if args.subjects and taken[exercise, label] > args.subjects:
            continue
        out = DEST / exercise / view / label / parts[-1]
        if out.exists() and out.stat().st_size == info.file_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(out, "wb") as dst:
            while chunk := src.read(1 << 22):
                dst.write(chunk)
        print("saved", out.relative_to(DEST.parent.parent), flush=True)


if __name__ == "__main__":
    main()
