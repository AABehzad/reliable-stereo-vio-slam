#pragma once

#include <vector>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/frontend/orb_extractor.hpp"

namespace my_slam
{

struct StereoLandmark
{
    cv::Point2f left_pixel;
    cv::Point2f right_pixel;

    cv::Point3d point_cam0_m;

    cv::DMatch match;

    double reprojection_error = 0.0;
};

struct StereoTriangulationResult
{
    bool success = false;

    Eigen::Matrix4d T_cam1_cam0 =
        Eigen::Matrix4d::Identity();

    double baseline_m = 0.0;

    std::size_t input_match_count = 0;
    std::size_t epipolar_inlier_count = 0;

    std::vector<StereoLandmark> landmarks;
};

class StereoGeometry
{
public:
    StereoTriangulationResult triangulate(
        const FeatureSet& left_features,
        const FeatureSet& right_features,
        const std::vector<cv::DMatch>& matches,
        const CameraModel& left_camera,
        const CameraModel& right_camera
    ) const;
};

} // namespace my_slam
