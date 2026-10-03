# Stereo VIO/SLAM Project — Tasks, Architecture, and Research Targets

## Executive Summary

This is a custom C++17 stereo VIO/SLAM system evaluated on EuRoC MH_01_easy.

The current architecture combines stereo calibration and triangulation, ORB matching, metric PnP, hybrid temporal/local-map tracking, constant-velocity prediction, reprojection gating, Pose Guard, KeyFrames, MapPoints, a seven-KeyFrame active LocalMap, Local Bundle Adjustment, loop-detection/pose-graph infrastructure, and a separate visual/IMU executable.

The main research target is Local BA landmark stability. Stage 15 introduced a pre-BA reliability score Q and then selective landmark optimization. The evidence shows that reliability weighting slightly improves trajectory metrics but does not prevent all pathological MapPoint motion. Selective optimization makes fixed landmarks numerically stationary and reduces large-movement counts, but extreme motion remains among jointly optimized landmarks.

## 1. Task History

### Repository audit

The audit task inspected Git state, source, CMake targets, build artifacts, result files, runtime logs, PnP, tracking, LocalMap, culling, Pose Guard, Local BA, loop closure, pose graph, and IMU paths. It also ran the 100-, 280-, 320-, and 500-frame executable checks.

The audit report is [PROJECT_STATUS_DELTA_FROM_HANDOFF.md](/workspace/PROJECT_STATUS_DELTA_FROM_HANDOFF.md).

Important limitation: SLAM_PROJECT_HANDOFF_CURRENT.md was not present under /workspace or /root, so exact handoff line citations and reliable handoff-to-source timing comparisons were unavailable.

### Stage 15: Active-Window-Aware Landmark Reliability

Implemented:

- pre-BA reliability score Q;
- observation, stereo, reprojection, and active-window terms;
- invalid-landmark quarantine;
- per-landmark scaled Huber loss;
- standard, reliability, and no-window modes;
- Q distributions and component diagnostics;
- landmark movement audits and Q-group diagnostics;
- mode-specific trajectory names.

The equation is:

    q_obs    = min(1, N_active / 3)
    q_stereo = N_stereo_active / max(1, N_active)
    q_reproj = exp(-0.5 * (median_preBA_reprojection / 2)^2)
    q_window = N_active / max(1, N_total)

    Q = 0.20 q_obs + 0.25 q_stereo
      + 0.35 q_reproj + 0.20 q_window

Q is clamped to [0.05, 1.0]. Ground truth and post-BA movement are not used.

### Stage 15 continuation: Selective Landmark Optimization

Mode 3 adds pre-BA landmark states:

- Q >= 0.60: JOINT_OPTIMIZE;
- 0.30 <= Q < 0.60: FIXED_LANDMARK;
- Q < 0.30: REJECTED;
- q_window < 0.35: forced fixed unless already rejected.

Fixed landmarks retain residuals but their MapPoint parameter blocks are constant. Rejected landmarks are excluded from this BA instance but remain in the global map. Transactional Ceres write-back remains guarded by summary.IsSolutionUsable().

## 2. Architecture

    EuRoC stereo images + calibration
                      |
                      v
            EuRoC Reader / Camera Model
                      |
                      v
            ORB extraction and matching
                      |
             +--------+--------+
             |                 |
             v                 v
       Stereo triangulation  Temporal matching
             |                 |
             +--------+--------+
                      |
                      v
             KeyFrames + MapPoints + LocalMap
                      |
                      v
             Hybrid correspondence builder
             - temporal candidates
             - active LocalMap candidates
             - temporal priority
             - one-to-one deduplication
                      |
                      v
             Motion prediction + reprojection gate
                      |
                      v
                    Metric PnP
                      |
                      v
                   Pose Guard
                      |
             +--------+--------+
             |                 |
             v                 v
       accepted pose      rejected pose stops safely
             |
             +-----------------------+
             |                       |
             v                       v
       inlier associations      KeyFrame decision
             |                       |
             +-----------+-----------+
                         |
                         v
                  Active LocalMap
                         |
                         v
                 Local Bundle Adjustment
                         |
             +-----------+-----------+
             |                       |
             v                       v
        optimized poses       optimized/fixed MapPoints
                         |
                         v
                  Loop / Pose Graph
                         |
                         v
                    optional IMU path

## 3. Project Targets

Primary research question:

Does active-window-aware reliability reduce pathological 3D MapPoint motion while preserving or improving trajectory accuracy?

Secondary research question:

Does q_window add value beyond observation count, stereo support, and reprojection consistency?

Engineering targets:

1. Keep tracking stable through difficult regions, especially Frame 274.
2. Prevent rejected PnP solutions from contaminating temporal state or the LocalMap.
3. Keep the active LocalMap bounded to seven KeyFrames.
4. Prevent weak or historically unsupported landmarks from destabilizing Local BA.
5. Preserve stereo residuals, Huber robustification, and transactional commits.
6. Keep runtime decisions independent of ground truth.
7. Produce valid trajectory, reprojection, and movement diagnostics.
8. Keep ROS outside the SLAM core.
9. Verify loop, pose-graph, and IMU paths separately.

## 4. Q Performance

Final Q distributions from the 0–100 experiments:

| Mode | Q mean | Q median | Q p10 | Q p25 | Q p75 | Q p90 | Quarantined |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.762701 | 0.791938 | 0.549767 | 0.703346 | 0.863614 | 0.866649 | 4 |
| Reliability | 0.762922 | 0.792347 | 0.549769 | 0.703685 | 0.863610 | 0.866648 | 4 |
| No-window | 0.791255 | 0.829096 | 0.597494 | 0.751480 | 0.862002 | 0.914884 | 4 |
| Selective | 0.759158 | 0.792154 | 0.528121 | 0.700897 | 0.863548 | 0.866644 | 0 |

The no-window distribution is higher because it omits the active-window penalty. This is expected and is not evidence that no-window landmarks are more reliable.

Final selective BA category counts:

| Category | Count |
|---|---:|
| Joint optimized | 1757 |
| Fixed landmarks | 458 |
| Rejected | 1 |
| Forced fixed by q_window | 166 |
| Quarantined | 0 |

## 5. 0–100 Performance Comparison

These results came from the Stage 15 experiments.

| Mode | ATE RMSE | RPE trans RMSE | RPE rot RMSE | Scale | Final error | Final reproj RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Standard joint BA | 0.016054 m | 0.007614 m | 0.079286 deg | 1.016299 | 0.019523 m | 0.708818 px |
| Reliability weighting | 0.015666 m | 0.007614 m | 0.078842 deg | 1.016772 | 0.019147 m | 0.708397 px |
| No-window ablation | 0.015457 m | 0.007604 m | 0.078647 deg | 1.016806 | 0.018758 m | 0.708444 px |
| Pose-only reference | 0.020100 m | 0.006056 m | 0.060813 deg | 0.961699 | 0.026980 m | 1.337100 px |
| Selective mode 3 | 0.015530 m | 0.005979 m | 0.060245 deg | 1.019726 | 0.017062 m | 0.771229 px |

Interpretation:

- mode 3 had the best final position error and best RPE among the joint BA modes in this comparison;
- mode 3 did not minimize reprojection error because it reduces landmark degrees of freedom;
- reprojection error is not a trajectory-accuracy guarantee;
- mode 3 improved trajectory behavior without eliminating all pathological MapPoint motion.

## 6. MapPoint Movement Performance

| Mode | Median | p95 | Maximum | >0.01 m | >0.10 m | >1 m | >10 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.007274 m | 0.472582 m | 1.04923e7 m | 13026 | 5254 | 640 | 78 |
| Reliability | 0.007161 m | 0.471105 m | 6.09909e6 m | 12987 | 5253 | 640 | 77 |
| No-window | 0.007031 m | 0.470093 m | 6.13645e6 m | 12963 | 5252 | 642 | 77 |
| Selective | 0.004507 m | 0.394084 m | 584021 m | 11845 | 4820 | 478 | 35 |

Mode 3 reduced the >10 m movement count by about 55% and the >1 m count by about 25% relative to standard joint BA. Its maximum movement remained pathological, so the method is an improvement, not a complete solution.

Selective category movement:

| Category | Audit records | Median | p95 | Maximum |
|---|---:|---:|---:|---:|
| Joint optimized | 25237 | 0.007591 m | 0.435883 m | 584021 m |
| Fixed landmarks | 2771 | 0 m | 0 m | 0 m |
| Rejected | 5 | 0 m | 0 m | 0 m |
| Forced fixed by q_window | 595 | 0 m | 0 m | 0 m |

The fixed-point result is exactly the intended mechanical behavior. Remaining extreme motion is concentrated in the joint category.

## 7. Tracking Robustness

| Run | Successful frames | KeyFrames | Total MPs | Active KFs | Active MPs | Avg inliers | Avg correspondences |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–100 | 100 | 18 | 2379 | 7 | 2216 | 515.12 | 524.82 |
| 0–280 | 280 | 47 | 3931 | 7 | 988 | 369.307 | 377.629 |
| 0–320 | 320 | 58 | 4538 | 7 | 1053 | 342.581 | 350.753 |
| 0–500 | 500 | 83 | 5196 | 7 | 696 | 347.404 | 354.022 |

Frame 274 remained stable:

- 61 PnP inliers;
- 66 guided correspondences;
- ratio 0.924242;
- prediction ON;
- 11 gated correspondences rejected;
- no Pose Guard rejection;
- processing continued through Frame 500.

## 8. Completed and Partial Work

Completed:

- EuRoC reader and calibration;
- ORB and descriptor matching;
- stereo triangulation and geometric filtering;
- metric PnP;
- KeyFrames, MapPoints, LocalMap, and culling;
- temporal/local hybrid tracking and deduplication;
- motion prediction and reprojection gating;
- Pose Guard;
- inlier-only state updates;
- Local BA and Stage 15 reliability modes;
- loop detector and pose-graph infrastructure;
- separate IMU propagation executable.

Partial or unverified:

- no geometrically verified loop occurred in the 500-frame run;
- pose graph was therefore not exercised on an accepted loop;
- IMU executable was not rerun during the audit;
- standalone geometry and PnP tests were not all rerun;
- Ceres produced step-evaluation warnings;
- the later 500-frame trajectory CSV was malformed/truncated for evaluation;
- pathological motion remains among jointly optimized landmarks.

## 9. Recommended Next Steps

1. Fix trajectory output integrity and add a row-count/stream-health assertion.
2. Re-run the evaluator after confirming complete CSVs.
3. Add a temporary-landmark-step limit or trust-region constraint.
4. Reject non-finite or implausible temporary Ceres point states before commit.
5. Instrument per-landmark depth, residual count, Q, state, and displacement.
6. Compare selective fixing against explicit landmark priors.
7. Keep Q unchanged while testing one new stabilization mechanism at a time.
8. Run the standalone stereo, triangulation, PnP, and IMU tests.
9. Verify loop closure and pose graph on a sequence containing an accepted loop.

## 10. Reproducibility

Build:

    cmake --build build

Stage 15 modes:

    ./build/run_keyframe_vo 0 100 1 0 0
    ./build/run_keyframe_vo 0 100 1 0 1
    ./build/run_keyframe_vo 0 100 1 0 2
    ./build/run_keyframe_vo 0 100 1 0 3
    ./build/run_keyframe_vo 0 100 1 1 0

Evaluator:

    python3 scripts/evaluate_vo.py /results/euroc_joint_ba_standard.csv
    python3 scripts/evaluate_vo.py /results/euroc_joint_ba_reliability.csv
    python3 scripts/evaluate_vo.py /results/euroc_joint_ba_no_window.csv
    python3 scripts/evaluate_vo.py /results/euroc_joint_ba_selective.csv
    python3 scripts/evaluate_vo.py /results/euroc_pose_only_reference.csv

Evidence is based on the current source tree, Git history through Stage 15, Stage 15 runtime logs, and the recorded 0–100 evaluator outputs. The handoff file is absent, and no claim is made that Q or selective optimization fully solves pathological landmark motion.

## 11. Lightweight Stage 15 Audit — Current Evaluation

This audit was limited to source inspection, a normal build, and fresh 0–100-frame runs. No production source was changed. The benchmark commands were:

    cmake --build build
    MY_SLAM_TRAJECTORY_PATH=/results/audit_stage15_standard_0_100.csv ./build/run_keyframe_vo 0 100 1 0 0
    MY_SLAM_TRAJECTORY_PATH=/results/audit_stage15_selective_0_100.csv ./build/run_keyframe_vo 0 100 1 0 3

Both runs exited successfully and produced 101-row trajectory files. The evaluator matched 79 poses to EuRoC ground truth; 22 initial poses were outside the ground-truth matching range.

### 11.1 Implementation verification

The reliability equation is implemented as specified:

- `LandmarkReliabilityMode::Selective = 3` and the optimization-state enum are declared in `include/my_slam/backend/local_bundle_adjuster.hpp:28-41`.
- The Q component fields and result counters are declared in `include/my_slam/backend/local_bundle_adjuster.hpp:43-90`.
- The configured scales and weights are in `src/backend/local_bundle_adjuster.cpp:35-47`.
- `q_obs`, `q_stereo`, `q_reproj`, `q_window`, the weighted sum, and the `[0.05, 1.0]` clamp are implemented in `src/backend/local_bundle_adjuster.cpp:871-938`.
- Selective partitioning is performed before residual construction in `src/backend/local_bundle_adjuster.cpp:1050-1076`: Q below 0.30 is rejected, Q below 0.60 is fixed, and otherwise `q_window < 0.35` forces the landmark fixed.
- Rejected landmarks are excluded from residual construction in `src/backend/local_bundle_adjuster.cpp:1152-1169`.
- Existing Huber plus reliability `ScaledLoss` residual handling is in `src/backend/local_bundle_adjuster.cpp:1370-1393`.
- Fixed MapPoints are made constant with `SetParameterBlockConstant` in `src/backend/local_bundle_adjuster.cpp:1458-1467`.
- Transactional write-back remains guarded by `summary.IsSolutionUsable()` in `src/backend/local_bundle_adjuster.cpp:1527-1535`.
- CLI mode parsing and mode-specific trajectory selection are in `apps/run_keyframe_vo.cpp:349-369` and `apps/run_keyframe_vo.cpp:499-513`.
- Runtime category movement diagnostics are emitted in `apps/run_keyframe_vo.cpp:2318-2342`.

The implementation therefore satisfies the requested Stage 15 semantics. The selective state decision is made from pre-BA geometry and observation support; it does not use ground truth or post-BA displacement.

### 11.2 Fresh 0–100 benchmark

| Mode | ATE RMSE | RPE trans RMSE | RPE rot RMSE | Scale | Final error | Final reproj RMSE | BA failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard joint BA | 0.016054 m | 0.007614 m | 0.079286 deg | 1.016299 | 0.019523 m | 0.708818 px | 0 |
| Selective Q BA | 0.015530 m | 0.005979 m | 0.060245 deg | 1.019726 | 0.017062 m | 0.771229 px | 0 |

Relative to standard joint BA, selective BA improved ATE by approximately 3.3%, translational RPE by 21.5%, rotational RPE by 24.0%, and final position error by 12.6%. Reprojection RMSE increased by approximately 8.8%, which is consistent with deliberately removing weak landmark degrees of freedom from joint optimization. The result supports improved trajectory stability, not a claim that selective BA minimizes image residuals.

### 11.3 Fresh MapPoint movement audit

| Mode | Median | P95 | Maximum | >0.01 m | >0.10 m | >1 m | >10 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard joint BA | 0.006682 m | 0.487572 m | 1.04923e7 m | 11824 | 4906 | 610 | 74 |
| Selective Q BA | 0.004507 m | 0.394084 m | 584021 m | 11845 | 4820 | 478 | 35 |

Selective BA reduced the P95 movement by 19.2%, the count above 1 m by 21.6%, and the count above 10 m by 52.7%. The maximum remains pathological and is concentrated in jointly optimized landmarks. The selective categories were:

| Category | Count | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Joint optimized | 1757 | 0.007591 m | 0.435883 m | 584021 m |
| Fixed landmarks | 458 | 0 m | 0 m | 0 m |
| Rejected | 1 | 0 m | 0 m | 0 m |
| Forced fixed by q_window | 166 | 0 m | 0 m | 0 m |

The zero movement of fixed and forced-fixed landmarks confirms the mechanical effect of the Ceres constant-block policy. It does not yet constrain the remaining jointly optimized outliers.

## 12. VIO-SLAM Readiness Assessment

### Implemented and structurally present

- Stereo EuRoC loading, calibration, camera transforms, triangulation, depth/reprojection checks, ORB matching, metric PnP, hybrid temporal/local-map tracking, motion prediction, reprojection gating, Pose Guard, KeyFrames, MapPoints, seven-KeyFrame LocalMap, culling, Local BA, Q reliability, and selective BA.
- Loop detection and pose-graph optimization are compiled into `my_slam_core`; loop verification uses descriptor matching, usable 3D–2D support, PnP inliers, inlier ratio, and relative-motion checks (`src/loop/loop_detector.cpp:36-203`). Pose-graph optimization uses odometry and loop edges with Huber losses and a transactional Ceres commit (`src/backend/pose_graph_optimizer.cpp:133-225`).
- IMU loading and calibration are implemented in `src/io/euroc_reader.cpp:156-257`.
- `apps/run_stereo_vio.cpp:120-256` performs IMU propagation, visual correction, and fused pose output. It initializes velocity from the first visual displacement, estimates gravity from early accelerometer samples, and keeps accelerometer/gyro biases at zero.

### Not yet sufficient for a complete tightly coupled VIO-SLAM claim

- The current IMU path is a separate propagation/correction executable, not a tightly coupled visual–inertial factor graph or joint Local BA. There is no production IMU residual/preintegration factor in `src/backend/local_bundle_adjuster.cpp`; the `src/imu` directory is present but contains no compiled implementation files in the current CMake target list.
- Biases are initialized to zero and bias optimization is explicitly disabled by `apps/run_stereo_vio.cpp:277-294`. Online bias estimation, calibrated noise handling in optimization, gravity refinement, and robust state covariance management remain.
- Loop and pose-graph infrastructure is present but not fully validated on an accepted loop. The 500-frame historical run tested candidates but accepted zero loops; this current audit intentionally did not repeat that long run.
- The prior audit found a malformed/truncated 500-frame trajectory output. The fresh 0–100 files were valid, but trajectory stream-health and row-count checks should still be made a permanent runtime invariant.
- Ceres emitted step-evaluation warnings in prior long runs. They were not application-level failures, but they reinforce the need for temporary-state validation and landmark-step protection.

### Concrete remaining stages

1. Add real IMU preintegration factors and state blocks for pose, velocity, gyro bias, accelerometer bias, and gravity to the backend.
2. Add visual–inertial joint optimization with calibrated noise/covariance, bias random-walk factors, and consistent camera–IMU extrinsics.
3. Add initialization observability checks for scale, gravity, velocity, and biases, then verify on EuRoC sequences with repeatable metrics.
4. Protect jointly optimized MapPoints with finite-state checks, depth bounds, a temporary step/trust-region limit, and a commit-time plausibility audit while keeping Q unchanged.
5. Repair and test trajectory output integrity with header/schema validation, row-count checks, flush/stream checks, and evaluator smoke tests.
6. Run the standalone stereo, triangulation, PnP, IMU, loop detector, and pose-graph tests; add regression coverage for Frame 274 and the selective category counts.
7. Validate a geometrically accepted loop on a sequence containing revisitation, then measure pose-graph corrections and map consistency.
8. Define release gates for full VIO-SLAM: no invalid states, bounded landmark motion, valid trajectories, repeatable ATE/RPE, verified loop closure, and tested IMU bias behavior.

## 13. Audit Conclusion

Stage 15 Q and selective landmark optimization are implemented correctly at the requested control points. The fresh 0–100 evidence shows that selective BA improves trajectory metrics and substantially reduces large MapPoint motion counts, while fixed landmarks remain numerically stationary. It is therefore a useful stabilization stage, but not a complete solution to pathological jointly optimized landmarks.

The project is best characterized as a strong stereo VO system with Local BA, experimental reliability-aware landmark control, loop/pose-graph infrastructure, and a loosely coupled IMU propagation path. The principal work required before claiming complete Stereo VIO-SLAM is tightly coupled IMU optimization, validated loop closure/pose graph behavior, stronger landmark-state safeguards, and reproducible runtime/evaluation integrity.
