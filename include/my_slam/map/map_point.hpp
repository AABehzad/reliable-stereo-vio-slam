#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <Eigen/Core>

namespace my_slam
{

using MapPointId = std::uint64_t;
using KeyFrameId = std::uint64_t;

struct Observation
{
    KeyFrameId keyframe_id = 0;
    std::size_t feature_index = 0;
};

struct MapPoint
{
    MapPointId id = 0;

    Eigen::Vector3d position_w =
        Eigen::Vector3d::Zero();

    bool active = true;

    std::unordered_map<
        KeyFrameId,
        Observation
    > observations;
};

} // namespace my_slam
