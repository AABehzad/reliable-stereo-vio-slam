#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

namespace my_slam
{

struct FeatureSet
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

class OrbExtractor
{
public:
    explicit OrbExtractor(
        int max_features = 1200,
        float scale_factor = 1.2f,
        int pyramid_levels = 8
    );

    FeatureSet extract(const cv::Mat& gray_image);

private:
    cv::Ptr<cv::ORB> orb_;
};

} // namespace my_slam