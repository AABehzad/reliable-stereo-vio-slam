#include "my_slam/map/local_map.hpp"

#include <stdexcept>

namespace my_slam
{

LocalMap::LocalMap(
    std::size_t max_keyframes
)
    : max_keyframes_(max_keyframes)
{
    if (max_keyframes_ < 2)
    {
        throw std::runtime_error(
            "LocalMap requires at least 2 keyframes."
        );
    }
}


std::shared_ptr<KeyFrame>
LocalMap::createKeyFrame(
    Timestamp timestamp_ns,
    const Eigen::Matrix4d& T_W_C,
    const FeatureSet& features
)
{
    auto keyframe =
        std::make_shared<KeyFrame>();

    keyframe->id =
        next_keyframe_id_++;

    keyframe->timestamp_ns =
        timestamp_ns;

    keyframe->T_W_C =
        T_W_C;

    keyframe->features =
        features;

    keyframes_[
        keyframe->id
    ] = keyframe;

    active_keyframes_.push_back(
        keyframe->id
    );

    enforceWindowSize();

    return keyframe;
}


std::shared_ptr<MapPoint>
LocalMap::createMapPoint(
    const Eigen::Vector3d& position_w
)
{
    auto point =
        std::make_shared<MapPoint>();

    point->id =
        next_map_point_id_++;

    point->position_w =
        position_w;

    map_points_[
        point->id
    ] = point;

    return point;
}


void LocalMap::addObservation(
    KeyFrameId keyframe_id,
    std::size_t feature_index,
    MapPointId map_point_id
)
{
    const auto kf =
        getKeyFrame(keyframe_id);

    const auto mp =
        getMapPoint(map_point_id);

    if (!kf || !mp)
    {
        throw std::runtime_error(
            "Invalid KeyFrame or MapPoint."
        );
    }

    Observation observation;

    observation.keyframe_id =
        keyframe_id;

    observation.feature_index =
        feature_index;

    mp->observations[
        keyframe_id
    ] = observation;

    kf->feature_to_mappoint[
        feature_index
    ] = map_point_id;
}


void LocalMap::addStereoObservation(
    KeyFrameId keyframe_id,
    std::size_t feature_index,
    MapPointId map_point_id,
    const Eigen::Vector2d& right_uv
)
{
    const auto kf = getKeyFrame(keyframe_id);
    const auto mp = getMapPoint(map_point_id);

    if (!kf || !mp)
    {
        throw std::runtime_error(
            "Invalid KeyFrame or MapPoint."
        );
    }

    Observation observation;
    observation.keyframe_id = keyframe_id;
    observation.feature_index = feature_index;
    observation.has_stereo = true;
    observation.right_uv = right_uv;

    mp->observations[keyframe_id] = observation;
    kf->feature_to_mappoint[feature_index] = map_point_id;
}


std::shared_ptr<KeyFrame>
LocalMap::getKeyFrame(
    KeyFrameId id
)
{
    const auto it =
        keyframes_.find(id);

    if (it == keyframes_.end())
    {
        return nullptr;
    }

    return it->second;
}


std::shared_ptr<MapPoint>
LocalMap::getMapPoint(
    MapPointId id
)
{
    const auto it =
        map_points_.find(id);

    if (it == map_points_.end())
    {
        return nullptr;
    }

    return it->second;
}


const std::deque<KeyFrameId>&
LocalMap::activeKeyFrames() const
{
    return active_keyframes_;
}


std::vector<
    std::shared_ptr<MapPoint>
>
LocalMap::activeMapPoints() const
{
    std::vector<
        std::shared_ptr<MapPoint>
    > result;

    for (const auto& [id, point] :
         map_points_)
    {
        if (!point->active)
        {
            continue;
        }

        bool observed_by_active_kf = false;

        for (const auto& kf_id :
             active_keyframes_)
        {
            if (point->observations.find(kf_id)
                != point->observations.end())
            {
                observed_by_active_kf = true;
                break;
            }
        }

        if (observed_by_active_kf)
        {
            result.push_back(point);
        }
    }

    return result;
}


std::size_t
LocalMap::keyFrameCount() const
{
    return keyframes_.size();
}


std::size_t
LocalMap::mapPointCount() const
{
    return map_points_.size();
}


void LocalMap::enforceWindowSize()
{
    while (
        active_keyframes_.size()
        > max_keyframes_
    )
    {
        active_keyframes_.pop_front();
    }
}


std::size_t
LocalMap::cullWeakMapPoints()
{
    std::size_t removed_count = 0;

    for (auto it = map_points_.begin();
         it != map_points_.end();)
    {
        const auto& map_point =
            it->second;

        bool visible_in_active_window =
            false;

        for (const auto keyframe_id :
             active_keyframes_)
        {
            if (map_point
                    ->observations
                    .find(keyframe_id)
                !=
                map_point
                    ->observations
                    .end())
            {
                visible_in_active_window =
                    true;

                break;
            }
        }

        // ----------------------------------------------
        // Still useful for local tracking / BA.
        // ----------------------------------------------

        if (visible_in_active_window)
        {
            map_point->active = true;

            ++it;

            continue;
        }

        map_point->active = false;

        // ----------------------------------------------
        // Point has already left the local window.
        //
        // If it was seen only once, it carries almost
        // no useful multi-view information.
        // ----------------------------------------------

        if (map_point->observations.size() <= 1)
        {
            it =
                map_points_.erase(it);

            ++removed_count;
        }
        else
        {
            // Multi-view landmark:
            // preserve it for later global-map /
            // loop-closure stages.
            ++it;
        }
    }

    return removed_count;
}

} // namespace my_slam
