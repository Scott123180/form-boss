# Form Boss Roadmap

This roadmap outlines the technical milestones, algorithmic fixes, and data collection needed to turn `form-boss` into a production-grade, gym-ready lift form analyzer.

---

## 🎯 High-Level Vision
Enable anyone with a smartphone in a real gym to record a set from any reasonable angle and receive accurate, actionable biomechanical coaching without false positives from setup movements, camera tilt, or gym bystanders.

---

## Phase 1: Real-World Rep Segmentation & Robustness (Immediate Priority)
*Observed gap: Unracking, racking, and floor setup create 0.2s–0.4s "ghost reps" with severe false depth/hinge faults.*

- [x] **Minimum Rep Duration Filter (`src/form_boss/reps.py`)** — done 2026-11-18
  - Per-lift `min_rep_s` / `min_phase_s` on `Lift`. Too-quick dips are reported as "ignored" (CLI, `report.json`, result page) instead of being dropped silently, and are removed before the remaining reps are re-measured so they no longer split a real rep.
  - Squat and bench: rep $\ge 0.6\text{s}$, each half $\ge 0.25\text{s}$. On the three user side-view squat clips this removes 13 setup movements (walk-in, ducking under the bar, walkout, racking) and keeps every working rep. The 1.0 s / 0.4 s first proposed would also drop real fast bodyweight squats in the Mendeley set (≈15%), and the "fast drop" tempo warning depends on keeping those.
  - Off for rows, deadlifts and the bodyweight/dumbbell lifts: real explosive barbell rows in `row_side_30fps_01` take ~0.5 s, and a deadlift dropped from lockout is lowered in ~0.3 s.
  - Slow setup movements (ducking under/out of the bar, walking the bar out) are left to **Outlier Dip Rejection** below.
- [x] **Active Set State Machine (`src/form_boss/sets.py`)** — done 2026-11-19
  - Reps are grouped into working sets by rhythm (a gap over max(15 s, 5 typical reps) starts a new set; the user squat clips have an 11 s breathing pause inside one set). The clip is split into `setup` → `working set` (→ `rest` → `working set` …) → `teardown`, shown in `report.json` (`sets`, `timeline`), the CLI and the result page ("Setup, before the set", "Resting between sets").
  - Only reps inside a working set are judged. Everything else is listed as not counted, with the reason: a lone dip before or after the set or between sets; a hold (3× the typical rep, or over 12 s); a foot moving to a new spot (walking the bar out or back; feet planted move <0.2 of the limit, walking 1–5×); the lift's joints out of the picture (walking up to the camera); out of the working position.
  - Working position, per lift and relative to the clip's own reps: torso hinged for the row (standing up between sets of rows was being judged as a "too upright" row: 9.7 s and 11.4 s "reps" in the two side-view clips), hands up on the bar for the squat (knee bends without the bar). If every rep is out of position they all stay in and get flagged, so a row done too upright is still called out.
  - The shallow-dip rule is applied again after these, so deep walking dips no longer inflate the "typical rep" and let a half-depth unrack through.
  - Result: exactly the 5 working reps on all five user squat clips (side ×3, front, rear); the front and rear clips had 0 and 12 before. Mendeley results are unchanged within ±1 person per lift.
  - Limits: the side-view rows reset the bar on the floor and the near plate hides the arm at the top, so some reps are still counted twice (no expected count in `tests/regression.toml` until that is fixed). The first rep's descent can include the walkout when the lifter never stands fully still in between (rear clip: 6.5 s).
- [x] **Outlier Dip Rejection** — done 2026-11-18
  - Per-lift `min_depth_frac` on `Lift`; on for squat and bench at 0.6. Dips shallower than that share of the clip's typical rep are reported as ignored ("too shallow for a rep").
  - The reference is the 75th percentile of the dips left after the duration filter, not the median: with 5 working reps and 2–3 half-depth setup dips the median is pulled down far enough that a 50%-of-median rule caught none of them.
  - Result on the three user side-view squat clips: exactly the 5 working reps each; the setup dips were 42–53% as deep. In the Mendeley squat clips the dips it removes are tracking errors (one-frame jumps under the face-blur box), not shallow reps.
  - Limits: a clip with a single dip has nothing to compare against, and a deliberately shallow rep in an otherwise deep set (below 60%) is listed as ignored rather than flagged for depth.

---

## Phase 2: Camera Pitch & Perspective Calibration
*Observed gap: Low floor-angle recordings cause horizontal thighs to appear tilted in 2D image coordinates, triggering false high-depth warnings.*

- [x] **Camera Pitch Estimation & Horizon Correction (`geometry.calibrate`)** — done 2026-11-19
  - From the frames where the lifter stands tall, the leg shows which way is up at the lifter (median natural lean on the Mendeley side views: 0.6 deg). In the picture that gives the camera's roll plus keystone lean; in the 3D estimate it also gives pitch.
  - Measurements are taken on a copy of the track turned so standing is vertical (when the roll is 2–15 deg) and with the 3D estimate turned so the leg points straight up; drawing uses the original. `report.json` → `camera`.
  - Setup advice when the camera is rolled (≥3 deg) or looking up from low down.
  - Pitch from the 3D estimate is rough and reads low: the user's floor-level clips, which look ~30 deg up, read +6 to +8; Mendeley's chest-height cameras, which look down, read −12 (side) to −18 (front). A pitch estimate from perspective (shins larger per metre than the torso when looking up) read +30 to +33 on the user side clips but swung ±30 deg on others, so it is not used.
  - **What the observed false "high" depth really was**: on the user side clips the rack's safety bar hides the hips at the bottom, and the tracker pins the hip to the top edge of the bar, ~12 cm too high (checked frame by frame; the 3D estimate inherits the same error). Correcting the 5–7 deg roll brings the readings from 17–29 deg above parallel to 12–22; the rest cannot be recovered from geometry. The setup advice now says to raise the camera to hip height, which also gets the safety bar out of the line of sight.
- [x] **Hybrid 3D/2D Fallback for Tilted Angles** — done 2026-11-19
  - From the side with |pitch| > 10 deg, thigh angle and depth come from the levelled 3D estimate. On the Mendeley side-view squats the good/bad gap in median bottom thigh angle grows from 5.0 deg (no correction) to 7.9 (roll + levelled 3D) to 9.1 (+ this fallback), and fewer "good" reps are flagged for depth (0.76 → 0.72).

---

## Phase 3: Performance & High-FPS Optimization
*Observed gap: 120fps iPhone clips run 15,000+ frames through MediaPipe CPU inference, taking minutes and producing bloated 200MB+ viewer files.*

Note: the two "120 fps" user clips are slow-motion clips exported at ~30 fps (3,794 and 3,137 frames, 29.8 fps average; only the container timebase says 120), so they were never 15,000 frames. The 200 MB came from the viewer transcode.

- [x] **Automatic Video Downsampling (`src/form_boss/pose.py`)** — done 2026-11-19
  - Footage at twice 30 fps or more is analysed 1 frame in n (60 → 2, 120 → 4, 240 → 8); 24–59 fps is kept as is. The page video and `--burn-in` overlay drop the same frames. Checked on a 60 fps copy of `squat_oblique_demo`: same 4 reps as the original.
  - Pose tracking is cached per clip in `out/<clip>/detections.npz` (keyed on file size, mtime and settings), so re-running `analyze` after changing a threshold takes seconds. `--retrack` ignores it.
- [x] **Viewer Transcode Efficiency (`src/form_boss/viewer.py`)** — done 2026-11-19
  - 1280 px on the long side, CRF 23, a keyframe every 0.5 s (was every 5 frames). The 101 s 1080p user squat clip goes from 159 MB to 30 MB; the 108 s rear clip is 27 MB.

---

## Phase 4: Video Dataset Collection (User Capture Tracker)
*Track of authentic gym videos needed to benchmark and tune each lift model.*

| Lift | Target Angles | Status | Needed Footage |
|---|---|---|---|
| **Squat** | Side (hip height), Front, Rear | 🟡 Partially Captured | Side **at hip height, clear of the safety bars** (the floor-level clips hide the hips at the bottom), low-bar vs high-bar comparison |
| **Barbell Row** | Side, Oblique, Front | 🟡 Captured | Strict pendlay row vs standard barbell row; side view from slightly in front so the plate does not hide the arm |
| **Bench Press** | Side (angled toward feet), Head-end | ⚪ Missing | Side view (unrack $\rightarrow$ touch $\rightarrow$ lockout), head-end view (bar level) |
| **Deadlift** | Side, 45° Oblique | ⚪ Missing | Conventional & Sumo deadlift from side (barbell over midfoot) |
| **Overhead Press** | Front, Side | ⚪ Missing | Strict standing barbell/dumbbell OHP |
| **Push-Up** | Side | ⚪ Missing | Standard push-ups with good lockout |
| **Bicep Curl** | Side | ⚪ Missing | Alternating dumbbell curls & barbell curl |
| **Triceps Extension** | Front, Side | ⚪ Missing | Overhead dumbbell/cable triceps extension |
| **Side Bend** | Front | ⚪ Missing | Standing dumbbell side bend |

---

## Phase 5: Multi-Angle Analysis & Exercise Expansion
*Observed gap: Frontal and rear views are currently stood down as "Not assessed: needs side view", missing major biomechanical cues.*

- [x] **Frontal & Rear Squat Checks** — done 2026-11-19
  - Reps are counted from hip height (as % of standing, against the standing height of the surrounding 8 s because the lifter walks toward and away from the camera), so front and rear clips are now analysed at all.
  - **Knee tracking**: knee width over ankle width from the bottom through the first half of the way up, corrected for perspective (from the front the knees are nearer the camera than the ankles at the bottom). The same knees-over-feet squats still read 0.97 from the front and 0.85 from behind, so only a clear cave (<0.8) is flagged.
  - **Hip shift**: hip centre against the feet's centre, % of stance width, against where it was before the rep. Separates best on the Mendeley front squats (flagged on 13% of "good" vs 28% of "bad" reps).
  - **Bar level**: wrist line against the line of the feet (so a crooked camera does not read as a tilted bar); only when the hands are on a bar across the shoulders.
  - Depth from the front is reported, not graded: hip height depends too much on camera height (the same squats read 50% from behind at knee height, 68% from in front at the floor). Hip/chest timing stands down from the front.
- [x] **Frontal Row Checks** — done 2026-11-19
  - **Pull symmetry** (elbow height difference at the top) and **forearm line** (forearm from vertical at the top). Reps counted from hand height. Forearm line separates the Mendeley front-view rows (bad flagged more for 12 people, good for 5); pull symmetry does not (5/5).
  - Hinge and torso movement stand down from the front: the 3D torso angle flagged 84% of "good" rows.
- [x] **Bystander & Multi-Person Filtering in Busy Gyms (`pose.pick_lifter`)** — done 2026-11-19
  - The tracked skeleton is split into tracklets wherever it jumps; the lifter is the tracklet with the largest summed body size, and other tracklets fill its gaps only if they stand where the lifter stands and are about the same size. On `row_oblique_30fps` this drops 18 s (44.8–63.3 s) where the tracker followed a man behind the lifter; those frames had been setting the view to "oblique 62 deg".
  - Tracking several people at once (MediaPipe `num_poses=3`) was tried and rejected: MediaPipe then re-detects people every frame instead of following the lifter, and the landmarks got ~8x noisier (the knee flipped at the bottom of every squat, splitting reps).
  - Setup advice when someone else was tracked for over 2% of the clip.
  - Limit: the lifter's frames while someone else was tracked are lost. The design for recovering them (repair pass on a crop, subject choice, person boxes) is in [docs/lifter-tracking.md](docs/lifter-tracking.md).
- [x] **Lift Auto-Detection (`lifts/detect.py`)** — done 2026-11-19
  - `--lift` defaults to `auto`. Rules on view-independent features: lying down (thigh direction in 3D), hips dropping, torso hinging, elbow bend while hinged, hand height, elbows staying up, sideways torso tilt.
  - All 16 regression clips; 374 of 414 Mendeley clips (90%; side and front views, all seven exercises). Weakest: front-view rows read as curls (the 3D estimate underestimates the hinge from the front), side bends with lots of forward fold.

---

## Phase 6: Codebase Health & Developer Experience

- [x] **Unit & Regression Test Suite** — done 2026-11-19
  - `tests/test_geometry.py` (angles, tilt, in-frame, local standing level, presets), `test_pose.py` (lifter picking on synthetic skeletons, frame stride, cache round trip), `test_sets.py` (state machine, holds, posture, walking), `test_reps.py`.
  - `scripts/regress.py` runs every clip in `tests/regression.toml` (user clips and samples; missing ones are skipped), checks the counted reps and diffs findings against `tests/regression_baseline.json` (`--update` to accept). `tests/test_regression.py` checks rep counts and lift detection in pytest from the cached tracking.
- [x] **Structured Logging & Error Handling** — done 2026-11-19
  - `form_boss.errors`: `VideoError`, `TrackingError`, `DependencyError`, `ConfigError` (all `FormBossError`); the CLI prints them as one line and exits 1. Progress goes through `logging` (`-v`, `-vv`, `-q`); the report itself is still printed.
- [x] **User Presets & Configuration** — done 2026-11-19
  - `--strictness lenient|normal|strict` scales how far from ideal a rep may be before it is flagged (×1.3 / ×1 / ×0.75 on each check's allowance).
  - `--variation`: squat `high_bar` (default), `low_bar` (more lean expected), `front` (less); deadlift `conventional`, `sumo` (shoulders may start further behind the bar); row `standard`, `pendlay` (back parallel, full stretch every rep). `form-boss lifts` lists them.
