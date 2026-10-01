#pragma once

#include <cstddef>
#include <vector>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/geometry/pnp_solver.hpp"
#include "my_slam/map/local_map.hpp"

namespace my_slam
{

struct LoopCandidate
{
    KeyFrameId current_keyframe_id = 0;
    KeyFrameId candidate_keyframe_id = 0;
    int descriptor_matches = 0;
    int usable_3d2d = 0;
    int geometric_inliers = 0;
    double inlier_ratio = 0.0;
    bool verified = false;
    Eigen::Matrix4d T_candidate_current =
        Eigen::Matrix4d::Identity();
};

using LoopConstraint = LoopCandidate;

struct LoopDetectionResult
{
    std::vector<LoopCandidate> candidates;
    std::vector<LoopCandidate> verified_loops;
    std::size_t historical_keyframes = 0;
    std::size_t queries = 0;
    std::size_t candidates_tested = 0;
    std::size_t rejected = 0;
    int best_descriptor_matches = 0;
    int best_geometric_inliers = 0;
};

class LoopDetector
{
public:
    LoopDetector(
        std::size_t minimum_separation = 10,
        std::size_t shortlist_size = 5
    );

    LoopDetectionResult detect(
        LocalMap& local_map,
        const std::shared_ptr<KeyFrame>& current,
        const CameraModel& camera
    ) const;

private:
    std::size_t minimum_separation_;
    std::size_t shortlist_size_;
};

} // namespace my_slam
