#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/map/local_map.hpp"

namespace my_slam
{

struct ResidualDistribution
{
    std::size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double p75 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double max = 0.0;
    std::size_t over_2px = 0;
    std::size_t over_3px = 0;
    std::size_t over_5px = 0;
};

enum class LandmarkReliabilityMode
{
    Standard = 0,
    ActiveWindow = 1,
    NoWindowAblation = 2,
    Selective = 3
};

enum class LandmarkOptimizationState
{
    JointOptimize = 0,
    FixedLandmark = 1,
    Rejected = 2
};

struct LandmarkReliability
{
    MapPointId id = 0;
    LandmarkOptimizationState state =
        LandmarkOptimizationState::JointOptimize;
    bool forced_fixed_by_q_window = false;
    double q = 0.0;
    double q_obs = 0.0;
    double q_stereo = 0.0;
    double q_reproj = 0.0;
    double q_window = 0.0;
    std::size_t n_total = 0;
    std::size_t n_active = 0;
    std::size_t n_stereo_active = 0;
};

struct LandmarkReliabilitySummary
{
    std::size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double p10 = 0.0;
    double p25 = 0.0;
    double p75 = 0.0;
    double p90 = 0.0;
    double mean_q_obs = 0.0;
    double mean_q_stereo = 0.0;
    double mean_q_reproj = 0.0;
    double mean_q_window = 0.0;
};

struct BundleAdjustmentResult
{
    bool success = false;

    std::size_t keyframes_optimized = 0;
    std::size_t map_points_optimized = 0;
    std::size_t observations_used = 0;
    std::size_t stereo_observations = 0;
    std::size_t mono_observations = 0;
    std::size_t skipped_invalid_depth = 0;
    std::size_t skipped_nonfinite = 0;
    std::size_t quarantined_landmarks = 0;
    std::size_t joint_optimized_landmarks = 0;
    std::size_t fixed_landmarks = 0;
    std::size_t rejected_landmarks = 0;
    std::size_t forced_fixed_by_q_window = 0;

    LandmarkReliabilitySummary reliability_summary;
    std::vector<LandmarkReliability> landmark_reliabilities;

    ResidualDistribution initial_cam0;
    ResidualDistribution final_cam0;
    ResidualDistribution initial_cam1;
    ResidualDistribution final_cam1;

    double initial_rmse_px = 0.0;
    double final_rmse_px = 0.0;

    double initial_cost = 0.0;
    double final_cost = 0.0;

    std::size_t iterations = 0;
};

class LocalBundleAdjuster
{
public:
    explicit LocalBundleAdjuster(
        std::size_t max_iterations = 50
    );

    BundleAdjustmentResult optimize(
        LocalMap& local_map,
        const CameraModel& cam0,
        const CameraModel& cam1,
        const Eigen::Matrix4d& T_cam1_cam0,
        bool optimize_points = true,
        LandmarkReliabilityMode reliability_mode =
            LandmarkReliabilityMode::Standard
    ) const;

private:
    std::size_t max_iterations_;
};

} // namespace my_slam
