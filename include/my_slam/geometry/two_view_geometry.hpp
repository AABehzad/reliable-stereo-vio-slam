#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/frontend/orb_extractor.hpp"

namespace my_slam
{

struct RelativePoseResult
{
    bool success = false;

    cv::Mat essential_matrix;
    cv::Mat rotation;
    cv::Mat translation;

    std::vector<cv::DMatch> inlier_matches;
};

class TwoViewGeometry
{
public:
    RelativePoseResult estimate(
        const FeatureSet& first,
        const FeatureSet& second,
        const std::vector<cv::DMatch>& matches,
        const CameraModel& camera
    ) const;
};

} // namespace my_slam
