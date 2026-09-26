#include "my_slam/frontend/local_map_tracker.hpp"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace my_slam
{

namespace
{

struct CandidateCorrespondence
{
    MapPointId map_point_id = 0;

    std::size_t current_feature_index = 0;

    float descriptor_distance = 0.0f;

    // 0 = temporal propagation from previous frame
    // 1 = active KeyFrame / local map
    int source_priority = 1;
};

} // namespace


LocalMapTracker::LocalMapTracker(
    float ratio_threshold
)
    : matcher_(ratio_threshold)
{
}


LocalMapTrackingResult
LocalMapTracker::buildCorrespondences(
    LocalMap& local_map,
    const FeatureSet& current_features,
    const FeatureSet& previous_features,
    const FrameMapAssociations&
        previous_feature_to_mappoint
) const
{
    LocalMapTrackingResult result;

    std::vector<CandidateCorrespondence>
        candidates;

    // ==================================================
    // 1. TEMPORAL PROPAGATION
    //
    // Previous frame -> current frame.
    //
    // These correspondences receive higher priority
    // because the previous frame is temporally closest
    // to the current frame.
    // ==================================================

    if (!previous_features.descriptors.empty() &&
        !current_features.descriptors.empty())
    {
        const auto temporal_matches =
            matcher_.match(
                previous_features,
                current_features
            );

        result.temporal_candidate_matches =
            temporal_matches.good_matches.size();

        for (const auto& match :
             temporal_matches.good_matches)
        {
            const std::size_t previous_feature_index =
                static_cast<std::size_t>(
                    match.queryIdx
                );

            const std::size_t current_feature_index =
                static_cast<std::size_t>(
                    match.trainIdx
                );

            const auto association =
                previous_feature_to_mappoint.find(
                    previous_feature_index
                );

            if (association ==
                previous_feature_to_mappoint.end())
            {
                continue;
            }

            const MapPointId map_point_id =
                association->second;

            const auto map_point =
                local_map.getMapPoint(
                    map_point_id
                );

            if (!map_point ||
                !map_point->active)
            {
                continue;
            }

            CandidateCorrespondence candidate;

            candidate.map_point_id =
                map_point_id;

            candidate.current_feature_index =
                current_feature_index;

            candidate.descriptor_distance =
                match.distance;

            candidate.source_priority = 0;

            candidates.push_back(
                candidate
            );
        }
    }

    // ==================================================
    // 2. ACTIVE LOCAL MAP
    //
    // Match all active KeyFrames to current frame.
    // ==================================================

    const auto& active_keyframes =
        local_map.activeKeyFrames();

    for (auto it =
             active_keyframes.rbegin();
         it != active_keyframes.rend();
         ++it)
    {
        const auto keyframe =
            local_map.getKeyFrame(*it);

        if (!keyframe)
        {
            continue;
        }

        ++result.active_keyframes_considered;

        const auto matches =
            matcher_.match(
                keyframe->features,
                current_features
            );

        result.local_candidate_matches +=
            matches.good_matches.size();

        for (const auto& match :
             matches.good_matches)
        {
            const std::size_t kf_feature_index =
                static_cast<std::size_t>(
                    match.queryIdx
                );

            const std::size_t current_feature_index =
                static_cast<std::size_t>(
                    match.trainIdx
                );

            const auto association =
                keyframe
                    ->feature_to_mappoint
                    .find(kf_feature_index);

            if (association ==
                keyframe
                    ->feature_to_mappoint
                    .end())
            {
                continue;
            }

            const MapPointId map_point_id =
                association->second;

            const auto map_point =
                local_map.getMapPoint(
                    map_point_id
                );

            if (!map_point ||
                !map_point->active)
            {
                continue;
            }

            CandidateCorrespondence candidate;

            candidate.map_point_id =
                map_point_id;

            candidate.current_feature_index =
                current_feature_index;

            candidate.descriptor_distance =
                match.distance;

            candidate.source_priority = 1;

            candidates.push_back(
                candidate
            );
        }
    }

    result.candidate_matches =
        result.temporal_candidate_matches
        +
        result.local_candidate_matches;

    // ==================================================
    // 3. Candidate ordering
    //
    // Temporal matches first.
    // Within each class, lower Hamming distance wins.
    // ==================================================

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const CandidateCorrespondence& a,
           const CandidateCorrespondence& b)
        {
            if (a.source_priority !=
                b.source_priority)
            {
                return
                    a.source_priority
                    <
                    b.source_priority;
            }

            return
                a.descriptor_distance
                <
                b.descriptor_distance;
        }
    );

    // ==================================================
    // 4. Deduplication
    //
    // One MapPoint can be used at most once.
    // One current feature can be used at most once.
    // ==================================================

    std::unordered_set<MapPointId>
        used_map_points;

    std::unordered_set<std::size_t>
        used_current_features;

    result.object_points.reserve(
        candidates.size()
    );

    result.image_points.reserve(
        candidates.size()
    );

    result.map_point_ids.reserve(
        candidates.size()
    );

    result.current_feature_indices.reserve(
        candidates.size()
    );

    for (const auto& candidate :
         candidates)
    {
        if (used_map_points.find(
                candidate.map_point_id)
            != used_map_points.end())
        {
            continue;
        }

        if (used_current_features.find(
                candidate.current_feature_index)
            != used_current_features.end())
        {
            continue;
        }

        const auto map_point =
            local_map.getMapPoint(
                candidate.map_point_id
            );

        if (!map_point ||
            !map_point->active)
        {
            continue;
        }

        const auto& p =
            map_point->position_w;

        result.object_points.emplace_back(
            static_cast<float>(p.x()),
            static_cast<float>(p.y()),
            static_cast<float>(p.z())
        );

        result.image_points.push_back(
            current_features
                .keypoints
                .at(
                    candidate.current_feature_index
                )
                .pt
        );

        result.map_point_ids.push_back(
            candidate.map_point_id
        );

        result.current_feature_indices.push_back(
            candidate.current_feature_index
        );

        used_map_points.insert(
            candidate.map_point_id
        );

        used_current_features.insert(
            candidate.current_feature_index
        );
    }

    return result;
}

} // namespace my_slam
