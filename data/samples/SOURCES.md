# Sample videos

Third-party clips used as test input. Not original work; keep attribution if redistributed.

| File | Source | Author | License |
|---|---|---|---|
| `squat_oblique_demo.webm` | https://commons.wikimedia.org/wiki/File:Barbell_back_squat_exercise_demonstration.webm | Your Move (https://ymove.app/exercise-api) | CC BY-SA 4.0 |
| `squat_rear_oblique_demo.webm` | https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm | see Commons page | CC BY 3.0 |

| `bench_side_demo.webm` | https://commons.wikimedia.org/wiki/File:Bench_press_-_exercise_demonstration_video.webm | FitnessScape | CC BY 3.0 |
| `bench_ymove_demo.webm` | https://commons.wikimedia.org/wiki/File:Barbell_bench_press_exercise_demonstration.webm | Your Move (https://ymove.app/exercise-api) | CC BY-SA 4.0 |
| `deadlift_side_demo.webm` | https://commons.wikimedia.org/wiki/File:Deadlift_-_exercise_demonstration_video.webm | FitnessScape | CC BY 3.0 |
| `deadlift_pesomuerto.webm` | https://commons.wikimedia.org/wiki/File:PesoMuerto.webm | Commons user Mariabeato4 | CC BY-SA 3.0 |
| `deadlift_sumo.webm` | https://commons.wikimedia.org/wiki/File:Peso_Muerto_sumo.webm | Commons user Martars022 | CC BY-SA 3.0 |

Note (2026-11-15): the two Your Move files are in the Commons category "Deletion requests November
2026", so their licence claims are under review there. Fine for local testing; check before publishing.

What the bench and deadlift clips show (2026-11-16): none is a true side view filmed with a still
camera, so they show the code runs and counts reps but do not test the thresholds.
`bench_side_demo` is filmed from about 40 deg off side-on, toward the feet. `bench_ymove_demo` is
filmed from the foot end, which counts as a front view. `deadlift_side_demo` is a rear oblique.
`deadlift_pesomuerto` and `deadlift_sumo` are 3D renders whose camera moves during the clip.
Two thresholds were changed after looking at these clips, both to stop false warnings: bench
lockout is now measured just outside the rep, and the bar-level warning starts at 8 deg instead of 5.

# Labelled evaluation clips (not stored in git)

`scripts/fetch_mendeley.py <exercise>` downloads clips into `data/mendeley/<exercise>/<view>/<good|bad>/`
from "A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels"
(https://data.mendeley.com/datasets/kgbb3yn47p, doi:10.17632/kgbb3yn47p.3, CC BY 4.0): about 25
people per exercise, one clip labelled good and one labelled bad each, from the side, front and
diagonal. `scripts/eval_mendeley.py` runs the analysis over them and writes `out/mendeley_eval/`.

The dataset's folder names are muscle groups; what the clips show, and the view each is evaluated from:

| Dataset folder | Movement | Lift | View |
|---|---|---|---|
| `squat` | bodyweight squat | `squat` | side |
| `push_up` | push-up | `push_up` | side |
| `bicep_curl` | standing dumbbell curl, mostly one arm | `bicep_curl` | side |
| `back` | bent-over dumbbell row | `bent_over_row` | side |
| `shoulder` | standing dumbbell overhead press | `overhead_press` | front |
| `tricep` | two-hand overhead dumbbell triceps extension | `triceps_extension` | front |
| `abs` | standing dumbbell side bend | `side_bend` | front |

Known problems with the set: some files are duplicates of each other across subjects or labels (the
eval drops them), "bad" is not broken down by fault type, the clips are 15 fps, the gym is busy, and
faces are blurred with large boxes that grow to cover the shoulders and arms and confuse the
tracker. Side-view triceps extensions mostly fail to track at all (arms and blur box hide the head;
in one clip the skeleton jumped to a seated bystander), which is why that lift is evaluated from the front.

## Results (2026-11-16)

For each person with both clips usable: is a larger share of reps flagged in their "bad" clip, in
their "good" clip, or about the same (within 0.2)? The labels were not used to build the checks,
except that the row's hinge and torso-movement thresholds were loosened after seeing they flagged
most "good" reps.

| Lift | People | Check that separates best | Bad more / good more / same | Any check: bad / good / same |
|---|---|---|---|---|
| squat | 16 | tempo; depth | 7/0/9; 5/1/10 | 0/0/16 |
| push_up | 18 | body line (bad clips pike or sag) | 9/1/8 | 9/1/8 |
| bicep_curl | 18 | top range; tempo | 9/1/8; 9/1/8 | 6/2/10 |
| bent_over_row | 20 | torso movement; pull height | 7/3/10; 5/1/14 | 6/3/11 |
| overhead_press | 25 | lockout; tempo | 12/3/10; 10/1/14 | 12/2/11 |
| triceps_extension | 21 | elbow flare | 13/1/7 | 12/2/7 |
| side_bend | 22 | range | 9/1/12 | 13/2/7 |

Reading this: the checks lean the right way on every lift but none is a classifier. Roughly half the
pairs come out "same", partly because many "good" clips have real faults (fast, shallow reps) and
partly from tracking noise. Weakest are the row (the blur box covers the bent-over upper body) and
the two estimates made without a direct view: forward fold in a side bend seen from the front, and
bottom range in a triceps extension seen from the front.

## Results (2026-11-19)

Same comparison after the working-set state machine, camera tilt correction, front/rear checks and
lift picking (see ROADMAP.md). Tracking is the cached tracking from before (`out/mendeley/.tracks`),
except for the front-view squat and row clips, fetched and tracked for this run.

| Lift | View | People | Check that separates best | Bad more / good more / same | Any check: bad / good / same |
|---|---|---|---|---|---|
| squat | side | 15 | tempo; depth | 6/0/9; 5/1/9 | 0/0/15 |
| squat | front | 23 | hip shift; tempo | 8/2/13; 9/4/10 | 11/1/11 |
| push_up | side | 17 | body line | 9/1/7 | 11/1/5 |
| bicep_curl | side | 18 | top range; tempo | 9/1/8; 9/1/8 | 6/2/10 |
| bent_over_row | side | 20 | torso movement | 7/3/10 | 6/4/10 |
| bent_over_row | front | 22 | forearm line | 12/5/5 | 9/2/11 |
| overhead_press | front | 24 | lockout; tempo | 11/3/10; 9/1/14 | 10/3/11 |
| triceps_extension | front | 21 | elbow flare | 13/1/7 | 12/2/7 |
| side_bend | front | 22 | range | 8/1/13 | 12/2/8 |

Side-view lifts are within one person of the 2026-11-16 run. On the side-view squats the median
bottom thigh angle now separates "good" from "bad" clips by 9.1 deg (was 5.0) after the camera
tilt correction. Knee tracking (front squat) rarely fires: the "bad" clips are not specifically
knee caves, and the threshold is set wide because perspective alone moves the reading by ~0.1.
Pull symmetry (front row) does not separate the labels (5/5/12).

Lift auto-detection on these clips: 374 of 414 (90%). Push-ups 42/42, curls 36/40, overhead press
48/50, side bends 43/46, triceps 47/50, squats 81/92, rows 77/94 (front-view rows are often read
as curls).
