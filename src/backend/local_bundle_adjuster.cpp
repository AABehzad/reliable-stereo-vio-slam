#include "my_slam/backend/local_bundle_adjuster.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace my_slam
{

namespace
{

struct PoseBlock
{
    double angle_axis[3] = {0.0, 0.0, 0.0};
    double translation[3] = {0.0, 0.0, 0.0};
};

struct PointBlock
{
    double xyz[3] = {0.0, 0.0, 0.0};
};


// ------------------------------------------------------------
// Distortion-aware reprojection residual.
//
// Pose parameters:
//     T_C_W = [R_CW | t_CW]
//
// Point:
//     X_W
//
// Camera:
//     pinhole + radial-tangential distortion
//
// distortion coefficients:
//     k1, k2, p1, p2
// ------------------------------------------------------------

struct ReprojectionCost
{
    ReprojectionCost(
        double u,
        double v,
        double fx,
        double fy,
        double cx,
        double cy,
        double k1,
        double k2,
        double p1,
        double p2
    )
        : u_(u),
          v_(v),
          fx_(fx),
          fy_(fy),
          cx_(cx),
          cy_(cy),
          k1_(k1),
          k2_(k2),
          p1_(p1),
          p2_(p2)
    {
    }

    template <typename T>
    bool operator()(
        const T* const camera_pose,
        const T* const point,
        T* residuals
    ) const
    {
        const T* rotation =
            camera_pose;

        const T* translation =
            camera_pose + 3;

        T point_camera[3];

        ceres::AngleAxisRotatePoint(
            rotation,
            point,
            point_camera
        );

        point_camera[0] +=
            translation[0];

        point_camera[1] +=
            translation[1];

        point_camera[2] +=
            translation[2];

        const T z =
            point_camera[2];

        // ----------------------------------------------------
        // Invalid depth.
        //
        // This normally gets filtered before the residual is
        // created. Keep a numerical fallback here as a final
        // safety measure.
        // ----------------------------------------------------

        const T safe_z =
            z > T(1e-8)
                ? z
                : T(1e-8);

        const T x =
            point_camera[0] /
            safe_z;

        const T y =
            point_camera[1] /
            safe_z;

        const T r2 =
            x * x +
            y * y;

        const T radial =
            T(1.0)
            +
            T(k1_) * r2
            +
            T(k2_) * r2 * r2;

        const T x_distorted =
            x * radial
            +
            T(2.0) * T(p1_) * x * y
            +
            T(p2_) *
                (r2 + T(2.0) * x * x);

        const T y_distorted =
            y * radial
            +
            T(p1_) *
                (r2 + T(2.0) * y * y)
            +
            T(2.0) * T(p2_) * x * y;

        const T projected_u =
            T(fx_) * x_distorted
            +
            T(cx_);

        const T projected_v =
            T(fy_) * y_distorted
            +
            T(cy_);

        residuals[0] =
            projected_u -
            T(u_);

        residuals[1] =
            projected_v -
            T(v_);

        return true;
    }

    double u_;
    double v_;

    double fx_;
    double fy_;
    double cx_;
    double cy_;

    double k1_;
    double k2_;
    double p1_;
    double p2_;
};


Eigen::Matrix4d
poseToCameraWorld(
    const PoseBlock& pose
)
{
    cv::Mat rvec(
        3,
        1,
        CV_64F
    );

    cv::Mat tvec(
        3,
        1,
        CV_64F
    );

    for (int i = 0; i < 3; ++i)
    {
        rvec.at<double>(i, 0) =
            pose.angle_axis[i];

        tvec.at<double>(i, 0) =
            pose.translation[i];
    }

    cv::Mat R;

    cv::Rodrigues(
        rvec,
        R
    );

    Eigen::Matrix4d T_C_W =
        Eigen::Matrix4d::Identity();

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            T_C_W(r, c) =
                R.at<double>(r, c);
        }

        T_C_W(r, 3) =
            tvec.at<double>(r, 0);
    }

    return T_C_W;
}


PoseBlock
cameraWorldToPose(
    const Eigen::Matrix4d& T_C_W
)
{
    PoseBlock pose;

    cv::Mat R(
        3,
        3,
        CV_64F
    );

    cv::Mat tvec(
        3,
        1,
        CV_64F
    );

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            R.at<double>(r, c) =
                T_C_W(r, c);
        }

        tvec.at<double>(r, 0) =
            T_C_W(r, 3);
    }

    cv::Mat rvec;

    cv::Rodrigues(
        R,
        rvec
    );

    for (int i = 0; i < 3; ++i)
    {
        pose.angle_axis[i] =
            rvec.at<double>(i, 0);

        pose.translation[i] =
            tvec.at<double>(i, 0);
    }

    return pose;
}


// ------------------------------------------------------------
// Read EuRoC distortion coefficients.
//
// CameraModel currently exposes distortion() as cv::Mat.
// Expected order for radial-tangential model:
//
// [k1, k2, p1, p2]
// ------------------------------------------------------------

bool getDistortionCoefficients(
    const CameraModel& camera,
    double& k1,
    double& k2,
    double& p1,
    double& p2
)
{
    const cv::Mat& d =
        camera.distortion();

    if (d.empty() ||
        d.total() < 4)
    {
        k1 = 0.0;
        k2 = 0.0;
        p1 = 0.0;
        p2 = 0.0;

        return false;
    }

    cv::Mat d64;

    d.convertTo(
        d64,
        CV_64F
    );

    k1 =
        d64.at<double>(0);

    k2 =
        d64.at<double>(1);

    p1 =
        d64.at<double>(2);

    p2 =
        d64.at<double>(3);

    return true;
}


// ------------------------------------------------------------
// Compute actual distorted reprojection RMSE.
// This is intentionally identical in model to the Ceres
// residual, so Initial/Final RMSE are meaningful.
// ------------------------------------------------------------

double computeObservationRmse(
    LocalMap& local_map,
    const CameraModel& camera
)
{
    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;

    getDistortionCoefficients(
        camera,
        k1,
        k2,
        p1,
        p2
    );

    double squared_error_sum = 0.0;
    std::size_t count = 0;

    const auto active_keyframes =
        local_map.activeKeyFrames();

    for (const auto keyframe_id :
         active_keyframes)
    {
        const auto keyframe =
            local_map.getKeyFrame(
                keyframe_id
            );

        if (!keyframe)
        {
            continue;
        }

        const Eigen::Matrix4d T_C_W =
            keyframe->T_W_C.inverse();

        for (const auto& [feature_index, map_point_id] :
             keyframe->feature_to_mappoint)
        {
            const auto map_point =
                local_map.getMapPoint(
                    map_point_id
                );

            if (!map_point ||
                !map_point->active)
            {
                continue;
            }

            if (feature_index >=
                keyframe
                    ->features
                    .keypoints
                    .size())
            {
                continue;
            }

            const Eigen::Vector4d X_W(
                map_point->position_w.x(),
                map_point->position_w.y(),
                map_point->position_w.z(),
                1.0
            );

            const Eigen::Vector4d X_C =
                T_C_W * X_W;

            if (!std::isfinite(
                    X_C.x()
                ) ||
                !std::isfinite(
                    X_C.y()
                ) ||
                !std::isfinite(
                    X_C.z()
                ) ||
                X_C.z() <= 1e-8)
            {
                continue;
            }

            const double x =
                X_C.x() /
                X_C.z();

            const double y =
                X_C.y() /
                X_C.z();

            const double r2 =
                x * x +
                y * y;

            const double radial =
                1.0
                +
                k1 * r2
                +
                k2 * r2 * r2;

            const double x_distorted =
                x * radial
                +
                2.0 * p1 * x * y
                +
                p2 *
                    (r2 + 2.0 * x * x);

            const double y_distorted =
                y * radial
                +
                p1 *
                    (r2 + 2.0 * y * y)
                +
                2.0 * p2 * x * y;

            const double u =
                camera.fx() *
                x_distorted
                +
                camera.cx();

            const double v =
                camera.fy() *
                y_distorted
                +
                camera.cy();

            const auto& observed =
                keyframe
                    ->features
                    .keypoints
                    [feature_index]
                    .pt;

            const double dx =
                u -
                static_cast<double>(
                    observed.x
                );

            const double dy =
                v -
                static_cast<double>(
                    observed.y
                );

            squared_error_sum +=
                dx * dx +
                dy * dy;

            ++count;
        }
    }

    if (count == 0)
    {
        return 0.0;
    }

    return std::sqrt(
        squared_error_sum /
        static_cast<double>(
            count
        )
    );
}

} // namespace


LocalBundleAdjuster::LocalBundleAdjuster(
    std::size_t max_iterations
)
    : max_iterations_(max_iterations)
{
}


BundleAdjustmentResult
LocalBundleAdjuster::optimize(
    LocalMap& local_map,
    const CameraModel& camera
) const
{
    BundleAdjustmentResult result;

    const auto active_keyframes =
        local_map.activeKeyFrames();

    const auto active_map_points =
        local_map.activeMapPoints();

    if (active_keyframes.size() < 2 ||
        active_map_points.empty())
    {
        return result;
    }

    result.keyframes_optimized =
        active_keyframes.size();

    // --------------------------------------------------------
    // Camera distortion.
    // --------------------------------------------------------

    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;

    const bool has_distortion =
        getDistortionCoefficients(
            camera,
            k1,
            k2,
            p1,
            p2
        );

    if (!has_distortion)
    {
        // Keep pinhole behaviour only when a camera truly has
        // no coefficients. For EuRoC this path is not expected.
        k1 = 0.0;
        k2 = 0.0;
        p1 = 0.0;
        p2 = 0.0;
    }

    // --------------------------------------------------------
    // Parameter blocks.
    //
    // Ceres uses T_C_W.
    // The map stores T_W_C.
    // --------------------------------------------------------

    std::unordered_map<
        KeyFrameId,
        PoseBlock
    > poses;

    std::unordered_map<
        MapPointId,
        PointBlock
    > points;

    poses.reserve(
        active_keyframes.size()
    );

    points.reserve(
        active_map_points.size()
    );

    for (const auto keyframe_id :
         active_keyframes)
    {
        const auto keyframe =
            local_map.getKeyFrame(
                keyframe_id
            );

        if (!keyframe)
        {
            continue;
        }

        const Eigen::Matrix4d T_C_W =
            keyframe->T_W_C.inverse();

        poses.emplace(
            keyframe_id,
            cameraWorldToPose(
                T_C_W
            )
        );
    }

    for (const auto& map_point :
         active_map_points)
    {
        if (!map_point)
        {
            continue;
        }

        PointBlock block;

        block.xyz[0] =
            map_point->position_w.x();

        block.xyz[1] =
            map_point->position_w.y();

        block.xyz[2] =
            map_point->position_w.z();

        points.emplace(
            map_point->id,
            block
        );
    }

    ceres::Problem problem;

    std::size_t observations_used = 0;

    // --------------------------------------------------------
    // Build reprojection observations.
    // --------------------------------------------------------

    for (const auto keyframe_id :
         active_keyframes)
    {
        const auto keyframe =
            local_map.getKeyFrame(
                keyframe_id
            );

        if (!keyframe)
        {
            continue;
        }

        const auto pose_it =
            poses.find(
                keyframe_id
            );

        if (pose_it == poses.end())
        {
            continue;
        }

        for (const auto& [feature_index, map_point_id] :
             keyframe->feature_to_mappoint)
        {
            const auto point_it =
                points.find(
                    map_point_id
                );

            if (point_it == points.end())
            {
                continue;
            }

            if (feature_index >=
                keyframe
                    ->features
                    .keypoints
                    .size())
            {
                continue;
            }

            const auto& map_point =
                local_map.getMapPoint(
                    map_point_id
                );

            if (!map_point)
            {
                continue;
            }

            // ----------------------------------------------
            // Exclude obviously invalid initial geometry.
            // ----------------------------------------------

            const Eigen::Matrix4d T_C_W =
                keyframe->T_W_C.inverse();

            const Eigen::Vector4d X_W(
                map_point->position_w.x(),
                map_point->position_w.y(),
                map_point->position_w.z(),
                1.0
            );

            const Eigen::Vector4d X_C =
                T_C_W * X_W;

            if (!X_C.allFinite() ||
                X_C.z() <= 1e-8)
            {
                continue;
            }

            const auto& keypoint =
                keyframe
                    ->features
                    .keypoints
                    [feature_index];

            auto* cost =
                new ceres::AutoDiffCostFunction<
                    ReprojectionCost,
                    2,
                    6,
                    3
                >(
                    new ReprojectionCost(
                        keypoint.pt.x,
                        keypoint.pt.y,
                        camera.fx(),
                        camera.fy(),
                        camera.cx(),
                        camera.cy(),
                        k1,
                        k2,
                        p1,
                        p2
                    )
                );

            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(1.0),
                pose_it->second.angle_axis,
                point_it->second.xyz
            );

            ++observations_used;
        }
    }

    result.observations_used =
        observations_used;

    if (observations_used < 10)
    {
        return result;
    }

    // --------------------------------------------------------
    // Gauge fixing.
    // --------------------------------------------------------

    const KeyFrameId fixed_keyframe_id =
        active_keyframes.front();

    const auto fixed_pose_it =
        poses.find(
            fixed_keyframe_id
        );

    if (fixed_pose_it != poses.end())
    {
        problem.SetParameterBlockConstant(
            fixed_pose_it
                ->second
                .angle_axis
        );
    }

    // --------------------------------------------------------
    // Initial reprojection error.
    // --------------------------------------------------------

    result.initial_rmse_px =
        computeObservationRmse(
            local_map,
            camera
        );

    // --------------------------------------------------------
    // Solver.
    // --------------------------------------------------------

    ceres::Solver::Options options;

    options.max_num_iterations =
        static_cast<int>(
            max_iterations_
        );

    options.linear_solver_type =
        ceres::SPARSE_SCHUR;

    options.minimizer_progress_to_stdout =
        false;

    options.num_threads =
        1;

    ceres::Solver::Summary summary;

    ceres::Solve(
        options,
        &problem,
        &summary
    );

    // --------------------------------------------------------
    // Write optimized poses back.
    // --------------------------------------------------------

    for (auto& [keyframe_id, pose] :
         poses)
    {
        const auto keyframe =
            local_map.getKeyFrame(
                keyframe_id
            );

        if (!keyframe)
        {
            continue;
        }

        const Eigen::Matrix4d
            T_C_W_optimized =
                poseToCameraWorld(
                    pose
                );

        keyframe->T_W_C =
            T_C_W_optimized.inverse();
    }

    // --------------------------------------------------------
    // Write optimized MapPoints back.
    // --------------------------------------------------------

    for (auto& [map_point_id, point] :
         points)
    {
        const auto map_point =
            local_map.getMapPoint(
                map_point_id
            );

        if (!map_point)
        {
            continue;
        }

        map_point->position_w =
            Eigen::Vector3d(
                point.xyz[0],
                point.xyz[1],
                point.xyz[2]
            );
    }

    // --------------------------------------------------------
    // Final reprojection error.
    // --------------------------------------------------------

    result.final_rmse_px =
        computeObservationRmse(
            local_map,
            camera
        );

    result.initial_cost =
        summary.initial_cost;

    result.final_cost =
        summary.final_cost;

    result.iterations =
        static_cast<std::size_t>(
            summary.iterations.size()
        );

    result.map_points_optimized =
        points.size();

    result.success =
        summary.IsSolutionUsable();

    return result;
}

} // namespace my_slam
