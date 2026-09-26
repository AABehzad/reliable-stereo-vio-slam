#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "my_slam/frontend/feature_matcher.hpp"
#include "my_slam/frontend/orb_extractor.hpp"
#include "my_slam/map/local_map.hpp"
#include "my_slam/map/map_point.hpp"

namespace my_slam
{

using FrameMapAssociations =
    std::unordered_map<std::size_t, MapPointId>;

struct LocalMapTrackingResult
{
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;

    // Aligned with object_points / image_points.
    std::vector<MapPointId> map_point_ids;

    std::vector<std::size_t>
        current_feature_indices;

    std::size_t active_keyframes_considered = 0;

    std::size_t temporal_candidate_matches = 0;
    std::size_t local_candidate_matches = 0;

    std::size_t candidate_matches = 0;
};

class LocalMapTracker
{
public:
    explicit LocalMapTracker(
        float ratio_threshold = 0.75f
    );

    LocalMapTrackingResult buildCorrespondences(
        LocalMap& local_map,
        const FeatureSet& current_features,
        const FeatureSet& previous_features,
        const FrameMapAssociations&
            previous_feature_to_mappoint
    ) const;

private:
    FeatureMatcher matcher_;
};

} // namespace my_slam
