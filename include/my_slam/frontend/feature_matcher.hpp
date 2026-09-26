#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

#include "my_slam/frontend/orb_extractor.hpp"

namespace my_slam
{

struct MatchResult
{
    std::vector<cv::DMatch> good_matches;

    std::size_t raw_match_count = 0;
};

class FeatureMatcher
{
public:
    explicit FeatureMatcher(
        float ratio_threshold = 0.75f
    );

    MatchResult match(
        const FeatureSet& first,
        const FeatureSet& second
    ) const;

private:
    float ratio_threshold_;
};

} // namespace my_slam