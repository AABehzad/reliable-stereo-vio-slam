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
        bool optimize_points = true
    ) const;

private:
    std::size_t max_iterations_;
};

} // namespace my_slam
