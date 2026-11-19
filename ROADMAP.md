# Form Boss Roadmap

This roadmap outlines the technical milestones, algorithmic fixes, and data collection needed to turn `form-boss` into a production-grade, gym-ready lift form analyzer.

---

## 🎯 High-Level Vision
Enable anyone with a smartphone in a real gym to record a set from any reasonable angle and receive accurate, actionable biomechanical coaching without false positives from setup movements, camera tilt, or gym bystanders.

---

## Phase 1: Real-World Rep Segmentation & Robustness (Immediate Priority)
*Observed gap: Unracking, racking, and floor setup create 0.2s–0.4s "ghost reps" with severe false depth/hinge faults.*

- [ ] **Minimum Rep Duration Filter (`src/form_boss/reps.py`)**
  - Enforce minimum total rep duration (e.g., $\ge 1.0\text{s}$) and minimum phase duration (descent $\ge 0.4\text{s}$, ascent $\ge 0.4\text{s}$).
  - Reject micro-dips from putting down or picking up the barbell.
- [ ] **Active Set State Machine**
  - Implement a three-phase detector: `SETUP / UNRACK` $\rightarrow$ `WORKING SET` $\rightarrow$ `RACK / TEARDOWN`.
  - Only evaluate form findings on reps inside the `WORKING SET` window.
  - Detect lifter standing upright between sets for rows/deadlifts so resting posture is not flagged as a form fault.
- [ ] **Outlier Dip Rejection**
  - Measure median dip amplitude across the clip; reject candidate reps whose amplitude is $< 50\%$ of the median (e.g., small knee flexes while adjusting stance).

---

## Phase 2: Camera Pitch & Perspective Calibration
*Observed gap: Low floor-angle recordings cause horizontal thighs to appear tilted in 2D image coordinates, triggering false high-depth warnings.*

- [ ] **Camera Pitch Estimation & Horizon Correction (`src/form_boss/geometry.py`)**
  - Estimate camera elevation angle / pitch from the vertical orientation of the lifter's shins or torso during the standing address.
  - Project 2D coordinates through an inverse perspective tilt transform before computing horizontal angles.
- [ ] **Hybrid 3D/2D Fallback for Tilted Angles**
  - When camera pitch exceeds $\pm 10^\circ$, use MediaPipe's gravity-aligned 3D world coordinates for `thigh_angle` and depth rather than raw screen pixel $y$-offsets.

---

## Phase 3: Performance & High-FPS Optimization
*Observed gap: 120fps iPhone clips run 15,000+ frames through MediaPipe CPU inference, taking minutes and producing bloated 200MB+ viewer files.*

- [ ] **Automatic Video Downsampling (`src/form_boss/pose.py`)**
  - Inspect input framerate: if video $\ge 45\text{ fps}$ (e.g., 60fps, 120fps, 240fps), temporally downsample to $\sim 30\text{ fps}$ during inference.
  - Speeds up pose tracking by **4×** with zero loss in biomechanical tracking fidelity.
- [ ] **Viewer Transcode Efficiency (`src/form_boss/viewer.py`)**
  - Cap transcoded `video.mp4` output at 30/60 fps and apply compression constraints (`-crf 23`, max width 1280px) to keep output bundles under 40 MB.

---

## Phase 4: Video Dataset Collection (User Capture Tracker)
*Track of authentic gym videos needed to benchmark and tune each lift model.*

| Lift | Target Angles | Status | Needed Footage |
|---|---|---|---|
| **Squat** | Side (hip height), Front, Rear | 🟡 Partially Captured | Side at hip height (bench/box mount), low-bar vs high-bar comparison |
| **Barbell Row** | Side, Oblique, Front | 🟡 Captured | Strict pendlay row vs standard barbell row |
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

- [ ] **Frontal & Rear Squat Checks**
  - **Knee Valgus (Inward Collapse)**: Track ratio of knee distance to ankle distance at the bottom of the squat.
  - **Lateral Hip Shift**: Detect asymmetry in horizontal hip center relative to the foot base.
  - **Barbell Level**: Measure barbell tilt across shoulders.
- [ ] **Frontal Row Checks**
  - **Pull Symmetry**: Measure bilateral elbow height and forearm alignment at the top of the row.
- [ ] **Bystander & Multi-Person Filtering in Busy Gyms**
  - Track the primary lifter using bounding box area and spatial continuity to prevent background gym-goers from hijacking the skeleton.
- [ ] **Lift Auto-Detection**
  - Allow omitting `--lift`; infer the lift type from the primary moving joint and trajectory.

---

## Phase 6: Codebase Health & Developer Experience

- [ ] **Unit & Regression Test Suite**
  - Unit tests for angle math, rep segmentation edge cases, and coordinate transforms.
  - Automated regression runner against `data/samples/` and `data/user/`.
- [ ] **Structured Logging & Error Handling**
  - Replace raw `SystemExit` and `print` statements with Python's standard `logging` module and custom exception classes.
- [ ] **User Presets & Configuration**
  - Support user-specified thresholds (e.g., `--variation low_bar`, `--strictness high`).
