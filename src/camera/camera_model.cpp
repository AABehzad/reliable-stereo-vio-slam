#include "my_slam/camera/camera_model.hpp"

#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace my_slam
{

CameraModel CameraModel::loadEuroc(
    const std::filesystem::path& sensor_yaml
)
{
    if (!std::filesystem::exists(sensor_yaml))
    {
        throw std::runtime_error(
            "Camera calibration file not found: "
            + sensor_yaml.string()
        );
    }

    const YAML::Node config =
        YAML::LoadFile(sensor_yaml.string());

    // --------------------------------------------------
    // Intrinsics
    // --------------------------------------------------

    if (!config["intrinsics"])
    {
        throw std::runtime_error(
            "Missing camera intrinsics."
        );
    }

    const auto intrinsics =
        config["intrinsics"]
            .as<std::vector<double>>();

    if (intrinsics.size() != 4)
    {
        throw std::runtime_error(
            "Expected intrinsics: fx, fy, cx, cy."
        );
    }

    CameraModel camera;

    camera.fx_ = intrinsics[0];
    camera.fy_ = intrinsics[1];
    camera.cx_ = intrinsics[2];
    camera.cy_ = intrinsics[3];

    camera.K_ =
        (cv::Mat_<double>(3, 3) <<
            camera.fx_, 0.0,        camera.cx_,
            0.0,        camera.fy_, camera.cy_,
            0.0,        0.0,        1.0);

    // --------------------------------------------------
    // Distortion
    // --------------------------------------------------

    if (config["distortion_coefficients"])
    {
        const auto coefficients =
            config["distortion_coefficients"]
                .as<std::vector<double>>();

        camera.distortion_ =
            cv::Mat(
                coefficients,
                true
            ).clone();
    }
    else
    {
        camera.distortion_ =
            cv::Mat::zeros(
                4,
                1,
                CV_64F
            );
    }

    // --------------------------------------------------
    // T_BS
    // --------------------------------------------------

    if (!config["T_BS"] ||
        !config["T_BS"]["data"])
    {
        throw std::runtime_error(
            "Missing T_BS in sensor.yaml."
        );
    }

    const auto transform_data =
        config["T_BS"]["data"]
            .as<std::vector<double>>();

    if (transform_data.size() != 16)
    {
        throw std::runtime_error(
            "T_BS must contain 16 values."
        );
    }

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            camera.T_BS_(row, col) =
                transform_data[row * 4 + col];
        }
    }

    return camera;
}

const cv::Mat& CameraModel::K() const
{
    return K_;
}

const cv::Mat& CameraModel::distortion() const
{
    return distortion_;
}

const Eigen::Matrix4d&
CameraModel::T_BS() const
{
    return T_BS_;
}

double CameraModel::fx() const
{
    return fx_;
}

double CameraModel::fy() const
{
    return fy_;
}

double CameraModel::cx() const
{
    return cx_;
}

double CameraModel::cy() const
{
    return cy_;
}

} // namespace my_slam
