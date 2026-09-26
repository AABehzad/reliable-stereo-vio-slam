#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "my_slam/camera/camera_model.hpp"

namespace my_slam
{

struct MetricPoseResult
{
    bool success = false;

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat rotation_matrix;

    int pnp_inlier_count = 0;

    std::vector<int> inlier_indices;
};

class PnPSolver
{
public:
    MetricPoseResult estimate(
        const std::vector<cv::Point3f>& object_points,
        const std::vector<cv::Point2f>& image_points,
        const CameraModel& camera
    ) const;

    MetricPoseResult estimateWithGuess(
        const std::vector<cv::Point3f>& object_points,
        const std::vector<cv::Point2f>& image_points,
        const CameraModel& camera,
        const cv::Mat& initial_rvec,
        const cv::Mat& initial_tvec
    ) const;

    MetricPoseResult estimateIterativeWithGuess(
        const std::vector<cv::Point3f>& object_points,
        const std::vector<cv::Point2f>& image_points,
        const CameraModel& camera,
        const cv::Mat& initial_rvec,
        const cv::Mat& initial_tvec
    ) const;
};

} // namespace my_slam
