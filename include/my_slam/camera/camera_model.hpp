#pragma once

#include <filesystem>

#include <Eigen/Core>

#include <opencv2/core.hpp>

namespace my_slam
{

class CameraModel
{
public:
    static CameraModel loadEuroc(
        const std::filesystem::path& sensor_yaml
    );

    const cv::Mat& K() const;
    const cv::Mat& distortion() const;

    const Eigen::Matrix4d& T_BS() const;

    double fx() const;
    double fy() const;
    double cx() const;
    double cy() const;

private:
    double fx_ = 0.0;
    double fy_ = 0.0;
    double cx_ = 0.0;
    double cy_ = 0.0;

    cv::Mat K_;
    cv::Mat distortion_;

    // EuRoC:
    // Transformation from sensor frame S
    // to body frame B.
    Eigen::Matrix4d T_BS_ =
        Eigen::Matrix4d::Identity();
};

} // namespace my_slam
