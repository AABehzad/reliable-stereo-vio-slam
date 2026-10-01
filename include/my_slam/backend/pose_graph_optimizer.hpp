#pragma once

#include <cstddef>
#include <vector>

#include "my_slam/loop/loop_detector.hpp"

namespace my_slam
{

enum class PoseGraphEdgeType
{
    ODOMETRY,
    LOOP
};

struct PoseGraphResult
{
    bool success = false;
    std::size_t nodes = 0;
    std::size_t odometry_edges = 0;
    std::size_t loop_edges = 0;
    std::size_t iterations = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    double mean_translation_correction = 0.0;
    double max_translation_correction = 0.0;
    double mean_rotation_correction_deg = 0.0;
    double max_rotation_correction_deg = 0.0;
    double trajectory_length_before = 0.0;
    double trajectory_length_after = 0.0;
};

class PoseGraphOptimizer
{
public:
    PoseGraphResult optimize(
        LocalMap& local_map,
        const std::vector<LoopConstraint>& constraints
    ) const;
};

} // namespace my_slam
