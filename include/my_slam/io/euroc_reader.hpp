#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "my_slam/core/types.hpp"

namespace my_slam
{

struct ImageRecord
{
    Timestamp timestamp_ns = 0;
    std::filesystem::path image_path;
};

struct ImuSample
{
    Timestamp timestamp_ns = 0;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
};

struct ImuCalibration
{
    double rate_hz = 0.0;
    double gyro_noise_density = 0.0;
    double gyro_random_walk = 0.0;
    double accel_noise_density = 0.0;
    double accel_random_walk = 0.0;
    Eigen::Matrix4d T_BS = Eigen::Matrix4d::Identity();
};

class EurocReader
{
public:
    explicit EurocReader(
        const std::filesystem::path& sequence_path
    );

    bool loadCam0();
    bool loadCam1();
    bool loadImu();

    const std::vector<ImageRecord>& cam0() const;
    const std::vector<ImageRecord>& cam1() const;
    const std::vector<ImuSample>& imu() const;
    const ImuCalibration& imuCalibration() const;

private:
    bool loadCamera(
        const std::string& camera_name,
        std::vector<ImageRecord>& output
    );

    std::filesystem::path sequence_path_;

    std::vector<ImageRecord> cam0_records_;
    std::vector<ImageRecord> cam1_records_;
    std::vector<ImuSample> imu_samples_;
    ImuCalibration imu_calibration_;
};

} // namespace my_slam
