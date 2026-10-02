#include "my_slam/backend/local_bundle_adjuster.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
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

struct ReliabilityConfig
{
    double obs_scale = 3.0;
    double reprojection_scale_px = 2.0;
    double min_q = 0.05;
    double huber_scale = 1.0;
    double obs_weight = 0.20;
    double stereo_weight = 0.25;
    double reprojection_weight = 0.35;
    double window_weight = 0.20;
};

constexpr ReliabilityConfig kReliabilityConfig{};

bool getDistortionCoefficients(
    const CameraModel& camera,
    double& k1,
    double& k2,
    double& p1,
    double& p2
);


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

        const T x =
            point_camera[0] /
            z;

        const T y =
            point_camera[1] /
            z;

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


struct StereoReprojectionCost
{
    StereoReprojectionCost(
        double u0,
        double v0,
        double u1,
        double v1,
        const CameraModel& cam0,
        const CameraModel& cam1,
        const Eigen::Matrix4d& T_cam1_cam0
    )
        : u0_(u0), v0_(v0), u1_(u1), v1_(v1),
          fx0_(cam0.fx()), fy0_(cam0.fy()),
          cx0_(cam0.cx()), cy0_(cam0.cy()),
          fx1_(cam1.fx()), fy1_(cam1.fy()),
          cx1_(cam1.cx()), cy1_(cam1.cy())
    {
        getDistortionCoefficients(
            cam0, k10_, k20_, p10_, p20_
        );
        getDistortionCoefficients(
            cam1, k11_, k21_, p11_, p21_
        );

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                rotation_3d_[r * 3 + c] =
                    T_cam1_cam0(r, c);
            }
            translation_[r] =
                T_cam1_cam0(r, 3);
        }
    }

    template <typename T>
    bool operator()(
        const T* const camera_pose,
        const T* const point,
        T* residuals
    ) const
    {
        T point_camera0[3];
        ceres::AngleAxisRotatePoint(
            camera_pose,
            point,
            point_camera0
        );

        point_camera0[0] += camera_pose[3];
        point_camera0[1] += camera_pose[4];
        point_camera0[2] += camera_pose[5];

        T point_camera1[3];
        for (int r = 0; r < 3; ++r)
        {
            point_camera1[r] =
                T(translation_[r]);
            for (int c = 0; c < 3; ++c)
            {
                point_camera1[r] +=
                    T(rotation_3d_[r * 3 + c])
                    * point_camera0[c];
            }
        }

        if (point_camera0[2] <= T(1e-8) ||
            point_camera1[2] <= T(1e-8))
        {
            return false;
        }

        project(
            point_camera0,
            fx0_, fy0_, cx0_, cy0_,
            k10_, k20_, p10_, p20_,
            residuals[0], residuals[1]
        );
        project(
            point_camera1,
            fx1_, fy1_, cx1_, cy1_,
            k11_, k21_, p11_, p21_,
            residuals[2], residuals[3]
        );

        residuals[0] -= T(u0_);
        residuals[1] -= T(v0_);
        residuals[2] -= T(u1_);
        residuals[3] -= T(v1_);
        return true;
    }

private:
    template <typename T>
    static void project(
        const T* point,
        double fx, double fy, double cx, double cy,
        double k1, double k2, double p1, double p2,
        T& u, T& v
    )
    {
        const T x = point[0] / point[2];
        const T y = point[1] / point[2];
        const T r2 = x * x + y * y;
        const T radial = T(1.0) + T(k1) * r2
            + T(k2) * r2 * r2;
        const T xd = x * radial + T(2.0) * T(p1) * x * y
            + T(p2) * (r2 + T(2.0) * x * x);
        const T yd = y * radial + T(p1) * (r2 + T(2.0) * y * y)
            + T(2.0) * T(p2) * x * y;
        u = T(fx) * xd + T(cx);
        v = T(fy) * yd + T(cy);
    }

    double u0_, v0_, u1_, v1_;
    double fx0_, fy0_, cx0_, cy0_;
    double fx1_, fy1_, cx1_, cy1_;
    double k10_ = 0.0, k20_ = 0.0, p10_ = 0.0, p20_ = 0.0;
    double k11_ = 0.0, k21_ = 0.0, p11_ = 0.0, p21_ = 0.0;
    double rotation_3d_[9] = {};
    double translation_[3] = {};
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


ResidualDistribution summarizeResiduals(
    std::vector<double> values
)
{
    ResidualDistribution result;

    if (values.empty())
    {
        return result;
    }

    std::sort(values.begin(), values.end());
    result.count = values.size();
    result.mean = std::accumulate(
        values.begin(), values.end(), 0.0
    ) / static_cast<double>(values.size());
    auto percentile = [&values](double q)
    {
        const auto index = static_cast<std::size_t>(
            q * static_cast<double>(values.size() - 1)
        );
        return values[index];
    };
    result.median = percentile(0.50);
    result.p75 = percentile(0.75);
    result.p90 = percentile(0.90);
    result.p95 = percentile(0.95);
    result.max = values.back();

    for (const double value : values)
    {
        if (value > 2.0) ++result.over_2px;
        if (value > 3.0) ++result.over_3px;
        if (value > 5.0) ++result.over_5px;
    }
    return result;
}


void projectDistorted(
    const Eigen::Vector4d& point_camera,
    const CameraModel& camera,
    cv::Point2d& pixel
)
{
    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    getDistortionCoefficients(camera, k1, k2, p1, p2);

    const double x = point_camera.x() / point_camera.z();
    const double y = point_camera.y() / point_camera.z();
    const double r2 = x * x + y * y;
    const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    const double xd = x * radial + 2.0 * p1 * x * y
        + p2 * (r2 + 2.0 * x * x);
    const double yd = y * radial + p1 * (r2 + 2.0 * y * y)
        + 2.0 * p2 * x * y;
    pixel.x = camera.fx() * xd + camera.cx();
    pixel.y = camera.fy() * yd + camera.cy();
}


void collectResiduals(
    LocalMap& local_map,
    const CameraModel& cam0,
    const CameraModel& cam1,
    const Eigen::Matrix4d& T_cam1_cam0,
    std::vector<double>& cam0_errors,
    std::vector<double>& cam1_errors
)
{
    for (const auto keyframe_id : local_map.activeKeyFrames())
    {
        const auto keyframe = local_map.getKeyFrame(keyframe_id);
        if (!keyframe) continue;
        const Eigen::Matrix4d T_C_W = keyframe->T_W_C.inverse();

        for (const auto& [feature_index, map_point_id] :
             keyframe->feature_to_mappoint)
        {
            const auto map_point = local_map.getMapPoint(map_point_id);
            if (!map_point || !map_point->active ||
                feature_index >= keyframe->features.keypoints.size())
            {
                continue;
            }

            const auto observation_it =
                map_point->observations.find(keyframe_id);
            if (observation_it == map_point->observations.end())
            {
                continue;
            }

            const Eigen::Vector4d X_W(
                map_point->position_w.x(),
                map_point->position_w.y(),
                map_point->position_w.z(), 1.0
            );
            const Eigen::Vector4d X_C0 = T_C_W * X_W;
            if (!X_C0.allFinite() || X_C0.z() <= 1e-8) continue;

            cv::Point2d projected;
            projectDistorted(X_C0, cam0, projected);
            const auto& left = keyframe->features.keypoints[feature_index].pt;
            cam0_errors.push_back(std::hypot(
                projected.x - static_cast<double>(left.x),
                projected.y - static_cast<double>(left.y)
            ));

            const auto& observation = observation_it->second;
            if (!observation.has_stereo ||
                !observation.right_uv.allFinite())
            {
                continue;
            }

            const Eigen::Vector4d X_C1 = T_cam1_cam0 * X_C0;
            if (!X_C1.allFinite() || X_C1.z() <= 1e-8) continue;
            projectDistorted(X_C1, cam1, projected);
            cam1_errors.push_back(std::hypot(
                projected.x - observation.right_uv.x(),
                projected.y - observation.right_uv.y()
            ));
        }
    }
}


struct ReliabilityObservation
{
    bool valid = false;
    bool stereo = false;
    bool active = false;
    double rmse = 0.0;
};

ReliabilityObservation evaluateReliabilityObservation(
    const MapPoint& map_point,
    KeyFrameId keyframe_id,
    const KeyFrame& keyframe,
    const CameraModel& cam0,
    const CameraModel& cam1,
    const Eigen::Matrix4d& T_cam1_cam0,
    const std::unordered_set<KeyFrameId>& active_ids
)
{
    ReliabilityObservation result;
    const auto observation_it = map_point.observations.find(keyframe_id);
    if (observation_it == map_point.observations.end()) return result;
    const auto& observation = observation_it->second;
    if (observation.feature_index >= keyframe.features.keypoints.size())
        return result;

    const Eigen::Vector4d X_W(
        map_point.position_w.x(), map_point.position_w.y(),
        map_point.position_w.z(), 1.0);
    const Eigen::Vector4d X_C0 = keyframe.T_W_C.inverse() * X_W;
    if (!X_C0.allFinite() || X_C0.z() <= 1e-8) return result;

    cv::Point2d left_projected;
    projectDistorted(X_C0, cam0, left_projected);
    if (!std::isfinite(left_projected.x) ||
        !std::isfinite(left_projected.y)) return result;
    const auto& left = keyframe.features.keypoints[observation.feature_index].pt;
    const double left_sq =
        std::pow(left_projected.x - static_cast<double>(left.x), 2.0) +
        std::pow(left_projected.y - static_cast<double>(left.y), 2.0);

    double squared_sum = left_sq;
    std::size_t component_count = 2;
    if (observation.has_stereo)
    {
        if (!observation.right_uv.allFinite()) return result;
        const Eigen::Vector4d X_C1 = T_cam1_cam0 * X_C0;
        if (!X_C1.allFinite() || X_C1.z() <= 1e-8) return result;
        cv::Point2d right_projected;
        projectDistorted(X_C1, cam1, right_projected);
        if (!std::isfinite(right_projected.x) ||
            !std::isfinite(right_projected.y)) return result;
        squared_sum +=
            std::pow(right_projected.x - observation.right_uv.x(), 2.0) +
            std::pow(right_projected.y - observation.right_uv.y(), 2.0);
        component_count = 4;
    }

    result.valid = true;
    result.stereo = observation.has_stereo;
    result.active = active_ids.count(keyframe_id) != 0;
    result.rmse = std::sqrt(squared_sum / static_cast<double>(component_count));
    return result;
}

std::vector<LandmarkReliability> computeLandmarkReliabilities(
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    LocalMap& local_map,
    const CameraModel& cam0,
    const CameraModel& cam1,
    const Eigen::Matrix4d& T_cam1_cam0,
    std::size_t& quarantined
)
{
    std::unordered_set<KeyFrameId> active_ids(
        local_map.activeKeyFrames().begin(), local_map.activeKeyFrames().end());
    std::vector<LandmarkReliability> result;
    quarantined = 0;
    for (const auto& point : map_points)
    {
        if (!point || !point->position_w.allFinite())
        {
            ++quarantined;
            continue;
        }
        LandmarkReliability item;
        item.id = point->id;
        std::vector<double> active_errors;
        for (const auto& [keyframe_id, unused] : point->observations)
        {
            const auto keyframe = local_map.getKeyFrame(keyframe_id);
            if (!keyframe) continue;
            const auto observation = evaluateReliabilityObservation(
                *point, keyframe_id, *keyframe, cam0, cam1,
                T_cam1_cam0, active_ids);
            if (!observation.valid) continue;
            ++item.n_total;
            if (observation.active)
            {
                ++item.n_active;
                if (observation.stereo) ++item.n_stereo_active;
                active_errors.push_back(observation.rmse);
            }
        }
        if (item.n_active == 0 || active_errors.empty())
        {
            ++quarantined;
            continue;
        }
        std::sort(active_errors.begin(), active_errors.end());
        const double median_reproj = active_errors[active_errors.size() / 2];
        if (!std::isfinite(median_reproj))
        {
            ++quarantined;
            continue;
        }
        item.q_obs = std::min(1.0, static_cast<double>(item.n_active) /
            kReliabilityConfig.obs_scale);
        item.q_stereo = static_cast<double>(item.n_stereo_active) /
            static_cast<double>(std::max<std::size_t>(1, item.n_active));
        item.q_reproj = std::exp(-0.5 * std::pow(
            median_reproj / kReliabilityConfig.reprojection_scale_px, 2.0));
        item.q_window = static_cast<double>(item.n_active) /
            static_cast<double>(std::max<std::size_t>(1, item.n_total));
        item.q = kReliabilityConfig.obs_weight * item.q_obs +
            kReliabilityConfig.stereo_weight * item.q_stereo +
            kReliabilityConfig.reprojection_weight * item.q_reproj +
            kReliabilityConfig.window_weight * item.q_window;
        item.q = std::clamp(item.q, kReliabilityConfig.min_q, 1.0);
        result.push_back(item);
    }
    return result;
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
    const CameraModel& cam0,
    const CameraModel& cam1,
    const Eigen::Matrix4d& T_cam1_cam0,
    bool optimize_points,
    LandmarkReliabilityMode reliability_mode
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
            cam0,
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

    result.landmark_reliabilities = computeLandmarkReliabilities(
        active_map_points, local_map, cam0, cam1, T_cam1_cam0,
        result.quarantined_landmarks);
    if (reliability_mode == LandmarkReliabilityMode::NoWindowAblation)
    {
        constexpr double remaining_weight =
            kReliabilityConfig.obs_weight +
            kReliabilityConfig.stereo_weight +
            kReliabilityConfig.reprojection_weight;
        for (auto& reliability : result.landmark_reliabilities)
        {
            reliability.q = std::clamp((
                kReliabilityConfig.obs_weight * reliability.q_obs +
                kReliabilityConfig.stereo_weight * reliability.q_stereo +
                kReliabilityConfig.reprojection_weight *
                    reliability.q_reproj) / remaining_weight,
                kReliabilityConfig.min_q, 1.0);
        }
    }
    result.reliability_summary.count =
        result.landmark_reliabilities.size();
    std::vector<double> q_values;
    q_values.reserve(result.landmark_reliabilities.size());
    for (const auto& reliability : result.landmark_reliabilities)
    {
        q_values.push_back(reliability.q);
        result.reliability_summary.mean += reliability.q;
        result.reliability_summary.mean_q_obs += reliability.q_obs;
        result.reliability_summary.mean_q_stereo += reliability.q_stereo;
        result.reliability_summary.mean_q_reproj += reliability.q_reproj;
        result.reliability_summary.mean_q_window += reliability.q_window;
    }
    if (!q_values.empty())
    {
        std::sort(q_values.begin(), q_values.end());
        const auto quantile = [&q_values](double q)
        {
            return q_values[static_cast<std::size_t>(
                q * static_cast<double>(q_values.size() - 1))];
        };
        result.reliability_summary.median = quantile(0.50);
        result.reliability_summary.p10 = quantile(0.10);
        result.reliability_summary.p25 = quantile(0.25);
        result.reliability_summary.p75 = quantile(0.75);
        result.reliability_summary.p90 = quantile(0.90);
        const double count = static_cast<double>(q_values.size());
        result.reliability_summary.mean /= count;
        result.reliability_summary.mean_q_obs /= count;
        result.reliability_summary.mean_q_stereo /= count;
        result.reliability_summary.mean_q_reproj /= count;
        result.reliability_summary.mean_q_window /= count;
    }

    std::unordered_map<MapPointId, double> landmark_q;
    landmark_q.reserve(result.landmark_reliabilities.size());
    for (const auto& reliability : result.landmark_reliabilities)
        landmark_q.emplace(reliability.id, reliability.q);

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

        if (landmark_q.find(map_point->id) == landmark_q.end())
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
    std::size_t stereo_observations = 0;
    std::size_t mono_observations = 0;

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

            if (!X_C.allFinite())
            {
                ++result.skipped_nonfinite;
                continue;
            }

            if (X_C.z() <= 1e-8)
            {
                ++result.skipped_invalid_depth;
                continue;
            }

            const auto observation_it =
                map_point->observations.find(
                    keyframe_id
                );

            if (observation_it ==
                map_point->observations.end())
            {
                ++result.skipped_nonfinite;
                continue;
            }

            const Observation& observation =
                observation_it->second;

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
                        cam0.fx(),
                        cam0.fy(),
                        cam0.cx(),
                        cam0.cy(),
                        k1,
                        k2,
                        p1,
                        p2
                    )
                );

            if (observation.has_stereo &&
                !observation.right_uv.allFinite())
            {
                ++result.skipped_nonfinite;
                delete cost;
                continue;
            }

            if (observation.has_stereo)
            {
                const Eigen::Vector4d X_C1 =
                    T_cam1_cam0 * X_C;

                if (!X_C1.allFinite())
                {
                    ++result.skipped_nonfinite;
                    delete cost;
                    continue;
                }

                if (X_C1.z() <= 1e-8)
                {
                    ++result.skipped_invalid_depth;
                    delete cost;
                    continue;
                }

                delete cost;

                auto* stereo_cost =
                    new ceres::AutoDiffCostFunction<
                        StereoReprojectionCost,
                        4, 6, 3
                    >(
                        new StereoReprojectionCost(
                            keypoint.pt.x,
                            keypoint.pt.y,
                            observation.right_uv.x(),
                            observation.right_uv.y(),
                            cam0,
                            cam1,
                            T_cam1_cam0
                        )
                    );

                const double q = landmark_q.at(map_point_id);
                ceres::LossFunction* loss = new ceres::HuberLoss(
                    kReliabilityConfig.huber_scale);
                if (reliability_mode != LandmarkReliabilityMode::Standard)
                    loss = new ceres::ScaledLoss(
                        loss, q, ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(
                    stereo_cost, loss, pose_it->second.angle_axis,
                    point_it->second.xyz
                );
                ++stereo_observations;
            }
            else
            {
                const double q = landmark_q.at(map_point_id);
                ceres::LossFunction* loss = new ceres::HuberLoss(
                    kReliabilityConfig.huber_scale);
                if (reliability_mode != LandmarkReliabilityMode::Standard)
                    loss = new ceres::ScaledLoss(
                        loss, q, ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(
                    cost, loss, pose_it->second.angle_axis,
                    point_it->second.xyz
                );
                ++mono_observations;
            }

            ++observations_used;
        }
    }

    result.observations_used =
        observations_used;
    result.stereo_observations =
        stereo_observations;
    result.mono_observations =
        mono_observations;

    if (observations_used < 10)
    {
        return result;
    }

    std::vector<double> initial_cam0_errors;
    std::vector<double> initial_cam1_errors;
    collectResiduals(
        local_map,
        cam0,
        cam1,
        T_cam1_cam0,
        initial_cam0_errors,
        initial_cam1_errors
    );
    result.initial_cam0 =
        summarizeResiduals(initial_cam0_errors);
    result.initial_cam1 =
        summarizeResiduals(initial_cam1_errors);

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

    if (!optimize_points)
    {
        for (auto& entry : points)
        {
            problem.SetParameterBlockConstant(
                entry.second.xyz
            );
        }
    }

    // --------------------------------------------------------
    // Initial reprojection error.
    // --------------------------------------------------------

    result.initial_rmse_px =
        computeObservationRmse(
            local_map,
            cam0
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
    // Solver result / transactional commit barrier.
    //
    // IMPORTANT:
    // Ceres optimizes temporary parameter storage. Do not
    // mutate LocalMap unless the solution is usable.
    // --------------------------------------------------------

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

    if (!result.success)
    {
        return result;
    }

    // --------------------------------------------------------
    // Write optimized poses back.
    // -------------------------------------------------------- --------------------------------------------------------

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

        if (optimize_points)
        {
            map_point->position_w =
                Eigen::Vector3d(
                    point.xyz[0],
                    point.xyz[1],
                    point.xyz[2]
                );
        }
    }

    // --------------------------------------------------------
    // Final reprojection error.
    // --------------------------------------------------------

    result.final_rmse_px =
        computeObservationRmse(
            local_map,
            cam0
        );

    std::vector<double> final_cam0_errors;
    std::vector<double> final_cam1_errors;
    collectResiduals(
        local_map,
        cam0,
        cam1,
        T_cam1_cam0,
        final_cam0_errors,
        final_cam1_errors
    );
    result.final_cam0 =
        summarizeResiduals(final_cam0_errors);
    result.final_cam1 =
        summarizeResiduals(final_cam1_errors);

    return result;
}

} // namespace my_slam
