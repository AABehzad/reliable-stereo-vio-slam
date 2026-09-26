#!/usr/bin/env python3

import csv
import sys
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DATASET = Path("/datasets/EuRoC/MH_01_easy")

VO_CSV = Path(
    sys.argv[1]
    if len(sys.argv) >= 2
    else "/results/euroc_metric_vo.csv"
)

GT_CSV = (
    DATASET
    / "mav0"
    / "state_groundtruth_estimate0"
    / "data.csv"
)

GT_SENSOR_YAML = (
    DATASET
    / "mav0"
    / "state_groundtruth_estimate0"
    / "sensor.yaml"
)

CAM0_SENSOR_YAML = (
    DATASET
    / "mav0"
    / "cam0"
    / "sensor.yaml"
)

REPORT_PATH = Path(
    "/results/euroc_vo_evaluation.txt"
)

GT_OUTPUT_CSV = Path(
    "/results/euroc_gt_matched.csv"
)

TRAJECTORY_PLOT = Path(
    "/results/euroc_vo_trajectory_3d.png"
)

ERROR_PLOT = Path(
    "/results/euroc_vo_position_error.png"
)


# ---------------------------------------------------------
# Quaternion / Rotation helpers
# Quaternion convention here:
# [w, x, y, z]
# ---------------------------------------------------------

def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)

    n = np.linalg.norm(q)

    if n < 1e-15:
        raise ValueError("Invalid zero quaternion.")

    return q / n


def quaternion_to_rotation(q):
    w, x, y, z = normalize_quaternion(q)

    return np.array([
        [
            1.0 - 2.0 * (y*y + z*z),
            2.0 * (x*y - z*w),
            2.0 * (x*z + y*w)
        ],
        [
            2.0 * (x*y + z*w),
            1.0 - 2.0 * (x*x + z*z),
            2.0 * (y*z - x*w)
        ],
        [
            2.0 * (x*z - y*w),
            2.0 * (y*z + x*w),
            1.0 - 2.0 * (x*x + y*y)
        ]
    ])


def slerp(q0, q1, alpha):
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = np.dot(q0, q1)

    # q and -q represent the same orientation.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return normalize_quaternion(q)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)

    theta = theta_0 * alpha

    s0 = (
        math.sin(theta_0 - theta)
        / sin_theta_0
    )

    s1 = (
        math.sin(theta)
        / sin_theta_0
    )

    return normalize_quaternion(
        s0 * q0 + s1 * q1
    )


def make_transform(position, quaternion_wxyz):
    T = np.eye(4, dtype=np.float64)

    T[:3, :3] = quaternion_to_rotation(
        quaternion_wxyz
    )

    T[:3, 3] = position

    return T


# ---------------------------------------------------------
# Parse EuRoC T_BS from sensor.yaml
#
# T_BS:
# sensor frame S -> body frame B
# ---------------------------------------------------------

def load_t_bs(path):
    text = Path(path).read_text(
        encoding="utf-8"
    )

    match = re.search(
        r"T_BS\s*:\s*.*?"
        r"data\s*:\s*\[([^\]]+)\]",
        text,
        flags=re.DOTALL
    )

    if match is None:
        raise RuntimeError(
            f"Could not parse T_BS from {path}"
        )

    values = [
        float(v.strip())
        for v
        in match.group(1).split(",")
    ]

    if len(values) != 16:
        raise RuntimeError(
            f"T_BS in {path} does not contain "
            f"16 values."
        )

    return np.array(
        values,
        dtype=np.float64
    ).reshape(4, 4)


# ---------------------------------------------------------
# VO trajectory
# ---------------------------------------------------------

def load_vo(path):
    frames = []
    timestamps = []
    transforms = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            frame = int(row["frame"])

            timestamp = int(
                row["timestamp_ns"]
            )

            p = np.array([
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"])
            ])

            # CSV:
            # qx,qy,qz,qw
            #
            # Internal:
            # qw,qx,qy,qz
            q = np.array([
                float(row["qw"]),
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"])
            ])

            T = make_transform(p, q)

            frames.append(frame)
            timestamps.append(timestamp)
            transforms.append(T)

    if not transforms:
        raise RuntimeError(
            "VO trajectory is empty."
        )

    return (
        np.asarray(frames),
        np.asarray(timestamps, dtype=np.int64),
        transforms
    )


# ---------------------------------------------------------
# EuRoC ground truth
#
# Relevant first columns:
#
# timestamp
# p_RS_R_x
# p_RS_R_y
# p_RS_R_z
# q_RS_w
# q_RS_x
# q_RS_y
# q_RS_z
# ---------------------------------------------------------

def load_ground_truth(path):
    timestamps = []
    positions = []
    quaternions = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.reader(f)

        for row in reader:

            if not row:
                continue

            if row[0].startswith("#"):
                continue

            timestamps.append(
                int(row[0])
            )

            positions.append([
                float(row[1]),
                float(row[2]),
                float(row[3])
            ])

            quaternions.append([
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7])
            ])

    if not timestamps:
        raise RuntimeError(
            "Ground truth is empty."
        )

    return (
        np.asarray(
            timestamps,
            dtype=np.int64
        ),
        np.asarray(
            positions,
            dtype=np.float64
        ),
        np.asarray(
            quaternions,
            dtype=np.float64
        )
    )


def interpolate_ground_truth(
    timestamp,
    gt_timestamps,
    gt_positions,
    gt_quaternions
):
    idx = np.searchsorted(
        gt_timestamps,
        timestamp
    )

    if idx == 0:
        if timestamp != gt_timestamps[0]:
            raise RuntimeError(
                "Requested timestamp precedes "
                "ground truth."
            )

        return (
            gt_positions[0],
            normalize_quaternion(
                gt_quaternions[0]
            )
        )

    if idx >= len(gt_timestamps):
        if timestamp != gt_timestamps[-1]:
            raise RuntimeError(
                "Requested timestamp exceeds "
                "ground truth."
            )

        return (
            gt_positions[-1],
            normalize_quaternion(
                gt_quaternions[-1]
            )
        )

    if gt_timestamps[idx] == timestamp:
        return (
            gt_positions[idx],
            normalize_quaternion(
                gt_quaternions[idx]
            )
        )

    i0 = idx - 1
    i1 = idx

    t0 = gt_timestamps[i0]
    t1 = gt_timestamps[i1]

    alpha = (
        float(timestamp - t0)
        / float(t1 - t0)
    )

    p = (
        (1.0 - alpha)
        * gt_positions[i0]
        +
        alpha
        * gt_positions[i1]
    )

    q = slerp(
        gt_quaternions[i0],
        gt_quaternions[i1],
        alpha
    )

    return p, q


def rotation_error_degrees(R):
    value = (
        np.trace(R) - 1.0
    ) / 2.0

    value = np.clip(
        value,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(value)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print(
        "===================================="
    )

    print(
        "      EuRoC VO Evaluation"
    )

    print(
        "===================================="
    )

    if not VO_CSV.exists():
        raise RuntimeError(
            f"VO file not found: {VO_CSV}"
        )

    # -----------------------------------------------------
    # Load VO
    # -----------------------------------------------------

    frames, vo_timestamps, T_est = (
        load_vo(VO_CSV)
    )

    print(
        f"VO poses: {len(T_est)}"
    )

    # -----------------------------------------------------
    # Load GT
    # -----------------------------------------------------

    (
        gt_timestamps,
        gt_positions,
        gt_quaternions
    ) = load_ground_truth(GT_CSV)

    print(
        f"Ground-truth samples: "
        f"{len(gt_timestamps)}"
    )

    # -----------------------------------------------------
    # Keep only the time interval where VO and GT overlap.
    #
    # EuRoC ground truth may start slightly after the
    # camera stream, so early camera poses can legitimately
    # have no corresponding ground-truth sample.
    # -----------------------------------------------------

    overlap_mask = (
        (vo_timestamps >= gt_timestamps[0])
        &
        (vo_timestamps <= gt_timestamps[-1])
    )

    valid_indices = np.nonzero(
        overlap_mask
    )[0]

    if len(valid_indices) < 2:
        raise RuntimeError(
            "VO and ground truth do not have enough "
            "overlapping timestamps."
        )

    skipped_before = int(valid_indices[0])

    skipped_after = int(
        len(vo_timestamps)
        - valid_indices[-1]
        - 1
    )

    frames = frames[valid_indices]

    vo_timestamps = (
        vo_timestamps[valid_indices]
    )

    T_est = [
        T_est[i]
        for i in valid_indices
    ]

    print(
        f"Overlapping VO poses: {len(T_est)}"
    )

    print(
        f"Skipped VO poses before GT: "
        f"{skipped_before}"
    )

    print(
        f"Skipped VO poses after GT: "
        f"{skipped_after}"
    )

    print(
        f"Evaluation timestamp range: "
        f"{vo_timestamps[0]} -> "
        f"{vo_timestamps[-1]}"
    )

    # -----------------------------------------------------
    # Rebase estimated trajectory to the first pose that
    # has valid ground truth.
    #
    # Original VO world frame is cam0 at the beginning of
    # the run. After dropping early poses, we want:
    #
    # first evaluated pose = Identity.
    #
    # T_ref_C(t) =
    # inverse(T_W_C_ref) * T_W_C(t)
    # -----------------------------------------------------

    T_W_C_ref = T_est[0]

    T_Cref_W = np.linalg.inv(
        T_W_C_ref
    )

    T_est = [
        T_Cref_W @ T
        for T in T_est
    ]

    # -----------------------------------------------------
    # Sensor extrinsics
    #
    # T_B_Sgt:
    # GT sensor -> body
    #
    # T_B_C0:
    # cam0 -> body
    # -----------------------------------------------------

    T_B_Sgt = load_t_bs(
        GT_SENSOR_YAML
    )

    T_B_C0 = load_t_bs(
        CAM0_SENSOR_YAML
    )

    T_Sgt_B = np.linalg.inv(
        T_B_Sgt
    )

    # -----------------------------------------------------
    # Generate GT cam0 pose at every VO timestamp.
    #
    # GT CSV supplies T_R_Sgt.
    #
    # Body:
    # T_R_B =
    # T_R_Sgt * T_Sgt_B
    #
    # cam0:
    # T_R_C0 =
    # T_R_B * T_B_C0
    # -----------------------------------------------------

    T_gt_absolute = []

    for timestamp in vo_timestamps:

        p, q = interpolate_ground_truth(
            timestamp,
            gt_timestamps,
            gt_positions,
            gt_quaternions
        )

        T_R_Sgt = make_transform(
            p,
            q
        )

        T_R_B = (
            T_R_Sgt
            @ T_Sgt_B
        )

        T_R_C = (
            T_R_B
            @ T_B_C0
        )

        T_gt_absolute.append(
            T_R_C
        )

    # -----------------------------------------------------
    # Normalize GT to initial cam0.
    #
    # Our VO world is:
    #
    # W = cam0 at start frame.
    #
    # So:
    #
    # T_Cstart_C(t) =
    # inverse(T_R_Cstart) * T_R_C(t)
    # -----------------------------------------------------

    T_R_C_start = (
        T_gt_absolute[0]
    )

    T_Cstart_R = np.linalg.inv(
        T_R_C_start
    )

    T_gt = [
        T_Cstart_R @ T
        for T in T_gt_absolute
    ]

    # -----------------------------------------------------
    # ATE position error
    #
    # IMPORTANT:
    # no similarity alignment
    # no scale correction
    # -----------------------------------------------------

    est_positions = np.asarray([
        T[:3, 3]
        for T in T_est
    ])

    gt_positions_local = np.asarray([
        T[:3, 3]
        for T in T_gt
    ])

    position_errors = np.linalg.norm(
        est_positions - gt_positions_local,
        axis=1
    )

    ate_rmse = math.sqrt(
        np.mean(
            position_errors ** 2
        )
    )

    ate_mean = np.mean(
        position_errors
    )

    ate_median = np.median(
        position_errors
    )

    ate_max = np.max(
        position_errors
    )

    # -----------------------------------------------------
    # RPE between consecutive VO samples
    # -----------------------------------------------------

    rpe_translation = []
    rpe_rotation_deg = []

    for i in range(
        len(T_est) - 1
    ):

        delta_est = (
            np.linalg.inv(T_est[i])
            @ T_est[i + 1]
        )

        delta_gt = (
            np.linalg.inv(T_gt[i])
            @ T_gt[i + 1]
        )

        delta_error = (
            np.linalg.inv(delta_gt)
            @ delta_est
        )

        rpe_translation.append(
            np.linalg.norm(
                delta_error[:3, 3]
            )
        )

        rpe_rotation_deg.append(
            rotation_error_degrees(
                delta_error[:3, :3]
            )
        )

    rpe_translation = np.asarray(
        rpe_translation
    )

    rpe_rotation_deg = np.asarray(
        rpe_rotation_deg
    )

    # -----------------------------------------------------
    # Path lengths
    # -----------------------------------------------------

    est_steps = np.diff(
        est_positions,
        axis=0
    )

    gt_steps = np.diff(
        gt_positions_local,
        axis=0
    )

    estimated_path_length = np.sum(
        np.linalg.norm(
            est_steps,
            axis=1
        )
    )

    gt_path_length = np.sum(
        np.linalg.norm(
            gt_steps,
            axis=1
        )
    )

    if gt_path_length > 1e-12:
        metric_scale_ratio = (
            estimated_path_length
            / gt_path_length
        )
    else:
        metric_scale_ratio = float(
            "nan"
        )

    # -----------------------------------------------------
    # Final displacement
    # -----------------------------------------------------

    est_final = est_positions[-1]
    gt_final = gt_positions_local[-1]

    final_position_error = np.linalg.norm(
        est_final - gt_final
    )

    # -----------------------------------------------------
    # Save matched GT
    # -----------------------------------------------------

    with open(
        GT_OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "timestamp_ns",
            "x_m",
            "y_m",
            "z_m"
        ])

        for frame, timestamp, p in zip(
            frames,
            vo_timestamps,
            gt_positions_local
        ):

            writer.writerow([
                int(frame),
                int(timestamp),
                p[0],
                p[1],
                p[2]
            ])

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    report = []

    report.append(
        "EuRoC Metric Stereo VO Evaluation"
    )

    report.append(
        "================================"
    )

    report.append(
        f"Number of poses: {len(T_est)}"
    )

    report.append("")

    report.append(
        "ATE position error "
        "(NO scale/alignment correction)"
    )

    report.append(
        f"ATE RMSE   : {ate_rmse:.6f} m"
    )

    report.append(
        f"ATE mean   : {ate_mean:.6f} m"
    )

    report.append(
        f"ATE median : {ate_median:.6f} m"
    )

    report.append(
        f"ATE max    : {ate_max:.6f} m"
    )

    report.append("")

    report.append(
        "RPE consecutive-frame error"
    )

    report.append(
        "RPE trans mean: "
        f"{np.mean(rpe_translation):.6f} m"
    )

    report.append(
        "RPE trans RMSE: "
        f"{math.sqrt(np.mean(rpe_translation**2)):.6f} m"
    )

    report.append(
        "RPE rot mean  : "
        f"{np.mean(rpe_rotation_deg):.6f} deg"
    )

    report.append(
        "RPE rot RMSE  : "
        f"{math.sqrt(np.mean(rpe_rotation_deg**2)):.6f} deg"
    )

    report.append("")

    report.append(
        f"Estimated path length: "
        f"{estimated_path_length:.6f} m"
    )

    report.append(
        f"GT path length       : "
        f"{gt_path_length:.6f} m"
    )

    report.append(
        f"Metric scale ratio   : "
        f"{metric_scale_ratio:.6f}"
    )

    report.append("")

    report.append(
        "Final estimated position:"
    )

    report.append(
        f"[{est_final[0]:.6f}, "
        f"{est_final[1]:.6f}, "
        f"{est_final[2]:.6f}] m"
    )

    report.append(
        "Final ground-truth position:"
    )

    report.append(
        f"[{gt_final[0]:.6f}, "
        f"{gt_final[1]:.6f}, "
        f"{gt_final[2]:.6f}] m"
    )

    report.append(
        f"Final position error: "
        f"{final_position_error:.6f} m"
    )

    report_text = "\n".join(
        report
    )

    print()
    print(report_text)

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8"
    )

    # -----------------------------------------------------
    # Plot trajectory
    # -----------------------------------------------------

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot(
        gt_positions_local[:, 0],
        gt_positions_local[:, 1],
        gt_positions_local[:, 2],
        label="Ground Truth"
    )

    ax.plot(
        est_positions[:, 0],
        est_positions[:, 1],
        est_positions[:, 2],
        label="Stereo VO"
    )

    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="o",
        label="Start"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title(
        "EuRoC Metric Stereo VO vs Ground Truth"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        TRAJECTORY_PLOT,
        dpi=180
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Position error plot
    # -----------------------------------------------------

    fig = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        frames,
        position_errors
    )

    plt.xlabel("Frame")
    plt.ylabel(
        "Position error [m]"
    )

    plt.title(
        "VO Position Error"
    )

    plt.grid(True)

    fig.tight_layout()

    fig.savefig(
        ERROR_PLOT,
        dpi=180
    )

    plt.close(fig)

    print()
    print(
        f"Report: {REPORT_PATH}"
    )

    print(
        f"Trajectory plot: "
        f"{TRAJECTORY_PLOT}"
    )

    print(
        f"Error plot: {ERROR_PLOT}"
    )

    print(
        f"Matched GT: "
        f"{GT_OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
