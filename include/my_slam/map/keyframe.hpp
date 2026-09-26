#pragma once

#include <cstdint>
#include <unordered_map>

#include <Eigen/Core>

#include "my_slam/core/types.hpp"
#include "my_slam/frontend/orb_extractor.hpp"
#include "my_slam/map/map_point.hpp"

namespace my_slam
{

struct KeyFrame
{
    KeyFrameId id = 0;

    Timestamp timestamp_ns = 0;

    // Camera -> world
    Eigen::Matrix4d T_W_C =
        Eigen::Matrix4d::Identity();

    FeatureSet features;

    // feature index -> MapPoint ID
    std::unordered_map<
        std::size_t,
        MapPointId
    > feature_to_mappoint;
};

} // namespace my_slam
