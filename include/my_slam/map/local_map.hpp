#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "my_slam/map/keyframe.hpp"
#include "my_slam/map/map_point.hpp"

namespace my_slam
{

class LocalMap
{
public:
    explicit LocalMap(
        std::size_t max_keyframes = 7
    );

    std::shared_ptr<KeyFrame>
    createKeyFrame(
        Timestamp timestamp_ns,
        const Eigen::Matrix4d& T_W_C,
        const FeatureSet& features
    );

    std::shared_ptr<MapPoint>
    createMapPoint(
        const Eigen::Vector3d& position_w
    );

    void addObservation(
        KeyFrameId keyframe_id,
        std::size_t feature_index,
        MapPointId map_point_id
    );

    void addStereoObservation(
        KeyFrameId keyframe_id,
        std::size_t feature_index,
        MapPointId map_point_id,
        const Eigen::Vector2d& right_uv
    );

    std::shared_ptr<KeyFrame>
    getKeyFrame(
        KeyFrameId id
    );

    std::shared_ptr<MapPoint>
    getMapPoint(
        MapPointId id
    );

    const std::deque<KeyFrameId>&
    activeKeyFrames() const;

    std::vector<
        std::shared_ptr<MapPoint>
    >
    activeMapPoints() const;

    std::size_t keyFrameCount() const;

    std::size_t mapPointCount() const;

    std::size_t cullWeakMapPoints();

private:
    void enforceWindowSize();

    std::size_t max_keyframes_;

    KeyFrameId next_keyframe_id_ = 0;
    MapPointId next_map_point_id_ = 0;

    std::unordered_map<
        KeyFrameId,
        std::shared_ptr<KeyFrame>
    > keyframes_;

    std::unordered_map<
        MapPointId,
        std::shared_ptr<MapPoint>
    > map_points_;

    std::deque<KeyFrameId>
        active_keyframes_;
};

} // namespace my_slam
