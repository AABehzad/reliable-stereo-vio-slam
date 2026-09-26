#include "my_slam/frontend/orb_extractor.hpp"

#include <stdexcept>

namespace my_slam
{

OrbExtractor::OrbExtractor(
    int max_features,
    float scale_factor,
    int pyramid_levels
)
{
    orb_ = cv::ORB::create(
        max_features,
        scale_factor,
        pyramid_levels
    );
}

FeatureSet OrbExtractor::extract(
    const cv::Mat& gray_image
)
{
    if (gray_image.empty())
    {
        throw std::runtime_error(
            "ORB extractor received an empty image."
        );
    }

    FeatureSet features;

    orb_->detectAndCompute(
        gray_image,
        cv::noArray(),
        features.keypoints,
        features.descriptors
    );

    return features;
}

} // namespace my_slam