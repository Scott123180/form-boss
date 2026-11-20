# form-boss

Lift form analysis from a phone video.

    uv sync
    uv run form-boss analyze data/samples/squat_oblique_demo.webm

Writes `out/<video name>/index.html` and opens it in your browser: the video with a
toggleable skeleton, bar/hand path and live labels, per-rep notes, camera-setup advice and
a scrubbable chart of the measurements. The page opens straight from disk. `report.json`
holds the same findings; `--burn-in` also writes `overlay.mp4` with the drawings baked in.

The lift is worked out from the movement; pass `--lift squat` (etc.) to set it. Other options:

    --variation low_bar        style of the lift (squat: high_bar, low_bar, front; deadlift:
                               conventional, sumo; row: standard, pendlay). `form-boss lifts` lists them
    --strictness strict        lenient | normal | strict: how far from ideal a rep may be before it is flagged
    --retrack                  ignore the cached pose tracking (out/<video name>/detections.npz)
    -v / -vv / -q              more or less progress output

## Lifts

Each lift is judged from the camera view its movement is visible in. The squat and row also have
checks for a front or rear view. Checks that need a different view are listed once as not assessed
instead of guessed at.

| `--lift` | Film from | A rep is counted from | Checks |
|---|---|---|---|
| `squat` | side; front or rear | knee bend (side), hip height (front/rear) | depth, hip/chest timing, torso lean, bar over midfoot, tempo; from the front or back: knee tracking (knees caving in), hip shift, bar level |
| `bench_press` | side, a step toward the feet | bar dropping from lockout | depth, lockout, touch point, bar over shoulders at lockout, hips on bench, bar level (head-end view), tempo |
| `deadlift` | side | hips opening from the floor position | hip/chest timing off the floor, start position, bar over midfoot and drift, lockout, leaning back, tempo |
| `push_up` | side | elbow bend | depth, lockout, body line (hip sag or pike), tempo |
| `bicep_curl` | side | elbow bend of whichever arm is curling | top range, bottom range, elbow position, body swing, tempo |
| `bent_over_row` | side; front | elbow bend (side), hands rising (front) | hinge angle, torso movement, pull height, bottom range, tempo; from the front: pull symmetry, forearm line |
| `overhead_press` | front | hands rising above the shoulders | lockout, bottom range, left/right symmetry, back lean (side view), tempo |
| `triceps_extension` | front | elbow bend | bottom range, lockout, elbow flare, upper arm and back arch (side view), tempo |
| `side_bend` | front | sideways torso lean, either direction | range, forward fold, hips over feet, tempo |

Angles are measured directly in the image when the joint moves in the image plane (a curl from
the side, a press from the front) and from the pose model's 3D estimate otherwise, which is less
exact. Where neither works, a view-independent stand-in is used, e.g. hand height above the
shoulder for a triceps extension seen from the front.

## How it works

Pipeline: pose tracking (`pose.py`: picks the lifter out from anyone else the tracker latches
onto, and analyses 60-240 fps footage at ~30 fps) -> body geometry corrected for camera tilt
(`geometry.py`) -> rep segmentation (`reps.py`) -> working sets (`sets.py`) -> lift-specific
signals, metrics and checks (`lifts/`) -> result page (`viewer.py`, `viewer.html`).

Only reps inside a working set are judged. Dips that are not reps are listed with the reason: too
quick (unracking, racking), much shallower than the clip's other reps (ducking under the bar),
walking (a foot moves to a new spot), out of the working position (standing up between sets of
rows), held too long, a lone movement before or after the set, or the joints out of the picture.

The camera's sideways tilt is measured from the standing lifter and corrected for; a camera looking
up from the floor gets setup advice, because equipment in front of the hips then hides them.

    uv run pytest                          # unit tests, plus regression checks on cached clips
    uv run python scripts/regress.py       # all clips in tests/regression.toml vs the saved baseline

A new lift is a module in `lifts/` registered in `lifts/__init__.py`. It supplies per-frame
signals, names the one that dips once per rep, turns each rep into measurements, and lists the
checks that turn measurements into findings (see `lifts/base.py`).

Known limits: one person is tracked at a time; a bystander who takes over is dropped afterwards,
but the lifter's frames during that time are lost. The bar or hand position is the midpoint of the
wrists; back rounding cannot be seen from shoulder and hip
points alone, so no lift checks it. Elbow flare on the bench can't be seen from the side or the head
end, so it isn't checked. From a true side view the near plate hides the hands and feet while a
deadlift bar is low, and the deadlift's start-position and bar-path checks stand down when that
happens. The camera has to stay still: the facing direction and camera tilt are worked out once
per clip. Hips hidden behind a rack's safety bars at the bottom of a squat are placed too high by
the tracker, which reads as a high squat; film from hip height.

Labelled test clips and how the checks do on them: `data/samples/SOURCES.md`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for what has been built, what the test footage showed, and the video
capture checklist.

