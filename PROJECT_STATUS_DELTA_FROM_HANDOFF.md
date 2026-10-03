# Project Status Delta from Handoff

## 1. Executive Summary

The repository is a working C++17 stereo visual-odometry/SLAM project with EuRoC loading, stereo calibration and triangulation, metric PnP, hybrid temporal/local-map tracking, constant-velocity prediction, reprojection gating, Pose Guard, KeyFrames, an active seven-KeyFrame LocalMap, Local Bundle Adjustment, loop-detection code, and pose-graph optimization.

The supplied handoff file \`SLAM_PROJECT_HANDOFF_CURRENT.md\` is not present in the repository or under \`/root\`; therefore its claims could not be checked against repository line ranges. The pasted handoff text was treated as contextual input only.

The current working tree is dirty because Stage 15 changes are uncommitted. The configured build succeeds. The requested 100-, 280-, 320-, and 500-frame \`run_keyframe_vo\` executions completed successfully. Frame 274 remained stable in the 280/320/500 runs; no Pose Guard rejection or catastrophic pose was observed there.

The current 500-frame trajectory file was malformed/truncated when passed to the evaluator, so no trustworthy 500-frame ATE/RPE result is claimed from that run.

## 2. Repository Identity

- Root: \`/workspace\`
- Branch: \`master\`
- HEAD: \`16cb34fc0d56fe36c2151c4b4c1fdd1192ac60b\` (\`stage 15\`)
- Previous commits: \`3d64ae7 stage 14\`, \`f1e91c7 Stage 14: Pose Graph Optimization for Stereo VIO-SLAM\`, \`7f75368 intializing using of imu\`, \`02576da initial\`
- Worktree: dirty
- Modified tracked files before this audit: \`apps/run_keyframe_vo.cpp\`, \`include/my_slam/backend/local_bundle_adjuster.hpp\`, \`src/backend/local_bundle_adjuster.cpp\`
- Handoff file: absent from \`/workspace\` and \`/root\`

## 3. Build Status

\`cmake --build build\` completed successfully with \`ninja: no work to do\`.

Configured primary targets include:

- \`my_slam_core\`
- \`run_keyframe_vo\`
- \`run_stereo_vio\`
- \`smoke_test\`

Existing build artifacts also include \`run_euroc\`, \`run_metric_vo\`, \`test_metric_pose\`, \`test_stereo\`, and \`test_triangulation\`.

## 4. Runtime Status

Dataset: \`/datasets/EuRoC/MH_01_easy\`.

| Run | Exit | Successful frames | KeyFrames | Total MPs | Active KFs | Active MPs | Avg PnP inliers | Avg map correspondences | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| \`0..100\` | 0 | 100 | 18 | 2379 | 7 | 2216 | 515.12 | 524.82 | \`/results/euroc_joint_ba_standard.csv\` |
| \`0..280\` | 0 | 280 | 47 | 3931 | 7 | 988 | 369.307 | 377.629 | \`/results/euroc_joint_ba_standard.csv\` |
| \`0..320\` | 0 | 320 | 58 | 4538 | 7 | 1053 | 342.581 | 350.753 | \`/results/euroc_joint_ba_standard.csv\` |
| \`0..500\` | 0 | 500 | 83 | 5196 | 7 | 696 | 347.404 | 354.022 | \`/results/euroc_joint_ba_standard.csv\` |

The executable logs \`localKF\`, \`temporalCand\`, \`localCand\`, \`candidates\`, \`mapCorr\`, \`inliers\`, \`ratio\`, and \`position\`. It also logs guided correspondence rejection and prediction status for Frames 268–280.

The logs contained Ceres warnings of the form \`Step failed to evaluate. Treating it as a step with infinite cost\`, but the application completed and reported \`Local BA completed\`; these are warnings, not application-level BA failures.

## 5. Last Confirmed Handoff State

The pasted handoff described a hybrid temporal/local-map tracker with a seven-KeyFrame local window and a known Frame 274 failure risk. Because \`SLAM_PROJECT_HANDOFF_CURRENT.md\` is absent, no exact handoff line ranges can be cited.

Confirmed directly in source and runtime:

- seven-KeyFrame active window;
- hybrid temporal plus active-local-map tracking;
- motion prediction and reprojection gating;
- PnP initial guess from the predicted pose;
- Pose Guard thresholds of \`0.35 m\` and \`15 deg\`;
- inlier-only association updates;
- safe failure before state/map/KeyFrame commit.

## 6. Current Repository State

### Confirmed implemented and verified in source/runtime

- EuRoC reader and camera calibration loading: \`src/io/euroc_reader.cpp\`, \`src/camera/camera_model.cpp\`; runtime loaded 3682 cam0/cam1 frames.
- Stereo transform convention: \`apps/run_keyframe_vo.cpp:425-443\`; computes \`inverse(T_BS_cam1) * T_BS_cam0\`.
- ORB extraction and descriptor matching: \`src/frontend/orb_extractor.cpp\`, \`src/frontend/feature_matcher.cpp\`.
- Stereo triangulation, positive-depth and reprojection filtering: \`src/geometry/stereo_geometry.cpp:66-110, 280-434\`.
- Metric PnP: \`src/geometry/pnp_solver.cpp:20-145\`.
- LocalMap and MapPoint/KeyFrame representations: \`src/map/local_map.cpp\`, \`include/my_slam/map/*.hpp\`.
- Seven-KeyFrame active window: \`apps/run_keyframe_vo.cpp:492-493\`, \`src/map/local_map.cpp:249-256\`.
- MapPoint culling: \`src/map/local_map.cpp:259-318\`; called at \`apps/run_keyframe_vo.cpp:1676-1677\`.
- Hybrid correspondence construction, temporal priority, deduplication, and one-to-one constraints: \`src/frontend/local_map_tracker.cpp:35-365\`.
- Constant-velocity prediction: \`apps/run_keyframe_vo.cpp:809-847\`.
- Reprojection gate: \`apps/run_keyframe_vo.cpp:831-1119\), gate threshold 30 px.
- PnP initial guess: \`apps/run_keyframe_vo.cpp:1174-1187\).
- Pose Guard: \`apps/run_keyframe_vo.cpp:1242-1295\`; rejection occurs before motion history, associations, KeyFrame creation, and map changes.
- Inlier-only current-frame associations: \`apps/run_keyframe_vo.cpp:1314-1355\`.
- Local Bundle Adjustment and transactional Ceres commit: \`src/backend/local_bundle_adjuster.cpp\`; commit is guarded by \`summary.IsSolutionUsable()\`.
- Loop detector and pose graph optimizer: \`src/loop/loop_detector.cpp\`, \`src/backend/pose_graph_optimizer.cpp\`; 500-frame runtime tested loop candidates but verified zero loops.

### Implemented but not runtime-verified in this audit

- IMU loading and visual/IMU propagation in \`apps/run_stereo_vio.cpp\` and \`src/io/euroc_reader.cpp\`. The requested audit runs used \`run_keyframe_vo\`.
- Standalone test executables such as \`test_stereo\`, \`test_triangulation\`, and \`test_metric_pose\`.

### Partially implemented or experimental

- Loop closure is implemented, but no loop was geometrically verified in the 500-frame run; pose-graph correction therefore did not execute.
- Local BA is implemented and invoked inline and once at the end, but Ceres emitted numerical step-evaluation warnings.
- The default trajectory path is mode-specific (\`/results/euroc_joint_ba_standard.csv\`), differing from the older \`/results/euroc_keyframe_vo.csv\` path.

## 7. Features Present in Source but Missing from Handoff

Timing relative to the absent handoff is unverified, but the current repository contains:

- constant-velocity prediction;
- predicted-pose PnP initialization;
- 30 px motion-guided reprojection gating;
- explicit Pose Guard implementation and diagnostics;
- active-map/temporal candidate deduplication and source priority;
- Local Bundle Adjustment;
- loop detector and pose-graph optimizer;
- IMU propagation/fusion executable;
- extensive runtime diagnostics and movement audits.

## 8. Features Documented in Handoff but Missing from Source

No repository copy of the handoff exists, so this category cannot be established against file text. From the pasted context alone, no confirmed absence was found for the explicitly listed motion-guided PnP items; they are implemented.

## 9. Features with Divergent Implementations

- The older handoff path was \`/results/euroc_keyframe_vo.csv\`; current \`run_keyframe_vo\` writes mode-specific paths.
- The source contains comments for “KeyFrame policy V1” and an implemented V2 policy. Active code uses V2 at \`apps/run_keyframe_vo.cpp:1397-1444\`.
- Historical MapPoints remain in the LocalMap while the active window is bounded to seven KeyFrames.

## 10. Post-Handoff Work — Confirmed

Git history shows:

- Stage 14 (\`3d64ae7\`, \`f1e91c7\`): loop detection and pose-graph infrastructure.
- Stage 15 (\`16cb34f\`): reliability-weighted Local BA changes are present in HEAD, with additional uncommitted modifications in the same three files.

Exact timing relative to the absent handoff remains unverified.

## 11. Post-Handoff Work — Timing Unverified

The following are present in the current tree but cannot be dated relative to the absent handoff:

- motion history and constant-velocity prediction;
- guided reprojection filtering;
- Pose Guard;
- hybrid correspondence merging and diagnostics;
- Local BA and landmark movement auditing;
- loop and pose graph execution paths;
- IMU visual propagation executable.

## 12. Motion-Guided PnP Status

Motion-Guided PnP is implemented:

1. Accepted pose history persists in \`T_W_C_prev_prev\` and \`T_W_C_prev\`.
2. A constant-velocity pose is extrapolated at \`apps/run_keyframe_vo.cpp:809-847\`.
3. Correspondences are filtered against the predicted pose using a 30 px reprojection gate.
4. PnP uses \`estimateIterativeWithGuess\` with predicted rotation/translation.
5. No unguided fallback is used after the gate rejects too many correspondences.

## 13. Frame 274 / Critical Region Status

Frame 274 is stable in the 280-, 320-, and 500-frame runs.

Observed 280-frame values:

- \`localKF=7\`
- \`temporalCand=668\`
- \`localCand=297\`
- \`candidates=965\`
- \`mapCorr=66\`
- \`inliers=61\`
- \`ratio=0.924242\`
- position \`[-0.173639, 0.105782, 0.101091] m\`
- guided rejection: 11 correspondences
- prediction: ON

No \`POSE GUARD rejected frame 274\` message occurred. Processing continued through Frame 500.

## 14. Runtime and Evaluation Results

Runtime logs:

- \`/tmp/slam_status_100.log\`
- \`/tmp/slam_status_280.log\`
- \`/tmp/slam_status_320.log\`
- \`/tmp/slam_status_500.log\`

The evaluator was invoked on \`/results/euroc_joint_ba_standard.csv\` after the 500-frame run, but failed because the trajectory file was malformed/truncated:

\`\`\`text
RuntimeError: VO trajectory is empty.
\`\`\`

A subsequent inspection found only 53 lines and a partial final row, producing:

\`\`\`text
TypeError: float() argument must be a string or a real number, not 'NoneType'
\`\`\`

No trustworthy ATE, RPE, scale, or final-position metric is claimed for these audit runs. The trajectory-output integrity issue must be fixed before evaluator metrics are used.

## 15. Remaining Work

1. Determine why the trajectory CSV is truncated despite the executable reaching its completion summary.
2. Re-run the evaluator after confirming a complete, parseable trajectory.
3. Add explicit trajectory row-count and stream-health checks.
4. Run standalone stereo, triangulation, metric-PnP, and IMU tests.
5. Investigate Ceres step-evaluation warnings and pathological MapPoint movement.
6. Verify loop closure and pose graph on a sequence containing a geometrically verifiable loop.
7. Document compatibility with the legacy trajectory filename if required downstream.

## 16. Recommended Next Step

Fix and instrument trajectory output first. The tracker survives the historical Frame 274 region, but the evaluator cannot consume the generated 500-frame trajectory. Until output integrity is resolved, trajectory metrics should not be used to judge further SLAM changes.

## 17. Evidence Table

| Item | Status | Evidence | File/Line or Symbol | Git Evidence | Runtime Evidence |
|---|---|---|---|---|---|
| EuRoC reader | Confirmed implemented and verified | Loads cam0/cam1 and dataset records | \`src/io/euroc_reader.cpp\` | Initial ancestry | 3682 frames loaded |
| Camera calibration | Confirmed implemented and verified | Loads both sensor YAML files | \`apps/run_keyframe_vo.cpp:425-443\` | HEAD | All runs started |
| Stereo transform | Confirmed implemented | \`inverse(T_BS_cam1) * T_BS_cam0\` | \`apps/run_keyframe_vo.cpp:441-443\` | HEAD | Stereo KeyFrames created |
| Active local map | Confirmed implemented and verified | \`LocalMap(7)\` and bounded deque | \`apps/run_keyframe_vo.cpp:492-493\` | HEAD | Active KFs remained 7 |
| Hybrid tracking | Confirmed implemented and verified | Temporal plus active-KF matching | \`src/frontend/local_map_tracker.cpp:35-223\` | HEAD | Candidate counters logged |
| Deduplication | Confirmed implemented | MapPoint/current-feature sets | \`src/frontend/local_map_tracker.cpp:224-365\` | HEAD | Consistency audit zeros |
| Inlier associations | Confirmed implemented | Only PnP inlier indices copied | \`apps/run_keyframe_vo.cpp:1314-1355\` | HEAD | Runs completed |
| Motion prediction | Confirmed implemented and verified | Constant-velocity extrapolation | \`apps/run_keyframe_vo.cpp:809-847\` | HEAD | Prediction ON at Frame 274 |
| Reprojection gate | Confirmed implemented and verified | 30 px gate | \`apps/run_keyframe_vo.cpp:831-1119\` | HEAD | Guided rejection counts |
| Pose Guard | Confirmed implemented and verified | 0.35 m / 15 deg gate | \`apps/run_keyframe_vo.cpp:1242-1303\` | HEAD | No rejection at Frame 274 |
| MapPoint culling | Confirmed implemented and verified | \`cullWeakMapPoints()\` called | \`src/map/local_map.cpp:259-318\` | HEAD | Culling counts logged |
| Local BA | Confirmed implemented and invoked | Ceres solve and transactional barrier | \`src/backend/local_bundle_adjuster.cpp\` | Stage 15 | Local BA completed |
| Loop closure | Partially implemented | Detector and verification thresholds | \`src/loop/loop_detector.cpp:36-203\` | Stage 14 | 255 tested, 0 verified |
| Pose graph | Implemented, not exercised successfully | Ceres graph optimizer | \`src/backend/pose_graph_optimizer.cpp:133-230\` | Stage 14 | No accepted loops |
| IMU | Implemented, not runtime-verified here | Reader and fusion executable | \`src/io/euroc_reader.cpp\`, \`apps/run_stereo_vio.cpp\` | \`7f75368\` ancestry | Not run |
| Evaluator output | Failed in audit | Generated CSV invalid | \`/results/euroc_joint_ba_standard.csv\` | N/A | Parse failure |

## 18. Exact Commands Executed

\`\`\`bash
pwd
git status --short --branch
git log --oneline --decorate --all -n 30
git reflog -n 15
find ...
cmake --build build --target help
cmake --build build
./build/run_keyframe_vo 0 100
./build/run_keyframe_vo 0 280
./build/run_keyframe_vo 0 320
./build/run_keyframe_vo 0 500
python3 scripts/evaluate_vo.py /results/euroc_joint_ba_standard.csv
\`\`\`

Runtime output was redirected to the four \`/tmp/slam_status_*.log\` files.

## 19. Limitations and Unknowns

- \`SLAM_PROJECT_HANDOFF_CURRENT.md\` is absent, so exact handoff line citations and direct file-to-file comparison are unavailable.
- Runtime metric evaluation failed because the current trajectory output is incomplete.
- Standalone tests and \`run_stereo_vio\` were not executed.
- The working tree contains uncommitted Stage 15 edits; this report does not attribute those edits to a specific commit beyond the repository’s Stage 15 history.
- Existing result files are historical and have mixed dates/names; they were not treated as proof of current-run behavior without matching logs.

