#include "my_slam/frontend/feature_matcher.hpp"

#include <stdexcept>

namespace my_slam
{

FeatureMatcher::FeatureMatcher(
    float ratio_threshold
)
    : ratio_threshold_(ratio_threshold)
{
}

MatchResult FeatureMatcher::match(
    const FeatureSet& first,
    const FeatureSet& second
) const
{
    if (first.descriptors.empty() ||
        second.descriptors.empty())
    {
        throw std::runtime_error(
            "FeatureMatcher received empty descriptors."
        );
    }

    cv::BFMatcher matcher(
        cv::NORM_HAMMING,
        false
    );

    std::vector<std::vector<cv::DMatch>> knn_matches;

    matcher.knnMatch(
        first.descriptors,
        second.descriptors,
        knn_matches,
        2
    );

    MatchResult result;

    result.raw_match_count =
        knn_matches.size();

    for (const auto& candidates : knn_matches)
    {
        if (candidates.size() < 2)
        {
            continue;
        }

        const cv::DMatch& best =
            candidates[0];

        const cv::DMatch& second_best =
            candidates[1];

        if (best.distance <
            ratio_threshold_ * second_best.distance)
        {
            result.good_matches.push_back(best);
        }
    }

    return result;
}

} // namespace my_slam