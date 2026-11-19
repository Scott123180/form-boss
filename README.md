# form-boss

Lift form analysis from a phone video.

    uv sync
    uv run form-boss analyze data/samples/squat_oblique_demo.webm --lift squat

Writes `out/<video name>/index.html` and opens it in your browser: the video with a
toggleable skeleton, bar/hand path and live labels, per-rep notes, camera-setup advice and
a scrubbable chart of the measurements. The page opens straight from disk. `report.json`
holds the same findings; `--burn-in` also writes `overlay.mp4` with the drawings baked in.

## Lifts

Each lift is judged from one camera view, the one its movement is visible in. Checks that need a
different view say so instead of guessing.

| `--lift` | Film from | A rep is counted from | Checks |
|---|---|---|---|
| `squat` | side | knee bend | depth, hip/chest timing, torso lean, bar over midfoot, tempo |
| `bench_press` | side, a step toward the feet | bar dropping from lockout | depth, lockout, touch point, bar over shoulders at lockout, hips on bench, bar level (head-end view), tempo |
| `deadlift` | side | hips opening from the floor position | hip/chest timing off the floor, start position, bar over midfoot and drift, lockout, leaning back, tempo |
| `push_up` | side | elbow bend | depth, lockout, body line (hip sag or pike), tempo |
| `bicep_curl` | side | elbow bend of whichever arm is curling | top range, bottom range, elbow position, body swing, tempo |
| `bent_over_row` | side | elbow bend | hinge angle, torso movement, pull height, bottom range, tempo |
| `overhead_press` | front | hands rising above the shoulders | lockout, bottom range, left/right symmetry, back lean (side view), tempo |
| `triceps_extension` | front | elbow bend | bottom range, lockout, elbow flare, upper arm and back arch (side view), tempo |
| `side_bend` | front | sideways torso lean, either direction | range, forward fold, hips over feet, tempo |

Angles are measured directly in the image when the joint moves in the image plane (a curl from
the side, a press from the front) and from the pose model's 3D estimate otherwise, which is less
exact. Where neither works, a view-independent stand-in is used, e.g. hand height above the
shoulder for a triceps extension seen from the front.

## How it works

Pipeline: pose tracking (`pose.py`) -> body geometry (`geometry.py`) -> rep
segmentation (`reps.py`) -> lift-specific signals, metrics and checks (`lifts/`) ->
result page (`viewer.py`, `viewer.html`).

A new lift is a module in `lifts/` registered in `lifts/__init__.py`. It supplies per-frame
signals, names the one that dips once per rep, turns each rep into measurements, and lists the
checks that turn measurements into findings (see `lifts/base.py`).

Known limits: one person is tracked, so bystanders can steal the skeleton in a busy gym; the bar
or hand position is the midpoint of the wrists; back rounding cannot be seen from shoulder and hip
points alone, so no lift checks it. Elbow flare on the bench can't be seen from the side or the head
end, so it isn't checked. From a true side view the near plate hides the hands and feet while a
deadlift bar is low, and the deadlift's start-position and bar-path checks stand down when that
happens. The camera has to stay still: the facing direction is worked out once per clip.

Labelled test clips and how the checks do on them: `data/samples/SOURCES.md`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the active development plan, planned algorithmic improvements, and video dataset capture checklist.

