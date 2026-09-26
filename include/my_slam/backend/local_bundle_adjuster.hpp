#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/map/local_map.hpp"

namespace my_slam
{

struct BundleAdjustmentResult
{
    bool success = false;

    std::size_t keyframes_optimized = 0;
    std::size_t map_points_optimized = 0;
    std::size_t observations_used = 0;

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
        const CameraModel& camera
    ) const;

private:
    std::size_t max_iterations_;
};

} // namespace my_slam
