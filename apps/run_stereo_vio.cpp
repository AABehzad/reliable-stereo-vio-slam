#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/io/euroc_reader.hpp"

namespace
{

struct PoseRecord
{
    my_slam::Timestamp timestamp = 0;
    Eigen::Matrix4d T_W_C = Eigen::Matrix4d::Identity();
};

std::vector<PoseRecord> loadTrajectory(const std::string& path)
{
    std::ifstream file(path);
    std::vector<PoseRecord> result;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == 'f') continue;
        std::stringstream stream(line);
        std::string field;
        std::vector<double> values;
        while (std::getline(stream, field, ','))
            values.push_back(std::stod(field));
        if (values.size() < 9) continue;
        Eigen::Quaterniond q(
            values[8], values[5], values[6], values[7]
        );
        if (!q.coeffs().allFinite())
            throw std::runtime_error("Invalid visual quaternion.");
        q.normalize();
        PoseRecord pose;
        pose.timestamp = static_cast<my_slam::Timestamp>(values[1]);
        pose.T_W_C = Eigen::Matrix4d::Identity();
        pose.T_W_C.block<3, 3>(0, 0) = q.toRotationMatrix();
        pose.T_W_C.block<3, 1>(0, 3) =
            Eigen::Vector3d(values[2], values[3], values[4]);
        result.push_back(pose);
    }
    return result;
}

void writePose(
    std::ofstream& file,
    std::size_t frame,
    const PoseRecord& visual,
    const Eigen::Matrix4d& T_W_C
)
{
    Eigen::Quaterniond q(T_W_C.block<3, 3>(0, 0));
    q.normalize();
    file << frame << ',' << visual.timestamp << ','
         << T_W_C(0, 3) << ',' << T_W_C(1, 3) << ','
         << T_W_C(2, 3) << ',' << q.x() << ',' << q.y()
         << ',' << q.z() << ',' << q.w() << '\n';
}

double rotationDifferenceDeg(
    const Eigen::Matrix3d& a,
    const Eigen::Matrix3d& b
)
{
    const Eigen::AngleAxisd difference(a.transpose() * b);
    return difference.angle() * 180.0 / 3.14159265358979323846;
}

bool validRotation(const Eigen::Matrix3d& rotation)
{
    return (rotation.transpose() * rotation -
            Eigen::Matrix3d::Identity()).norm() < 1e-6 &&
           std::abs(rotation.determinant() - 1.0) < 1e-6;
}

} // namespace

int main(int argc, char** argv)
{
    const std::size_t start = argc > 1 ? std::stoul(argv[1]) : 0;
    const std::size_t end = argc > 2 ? std::stoul(argv[2]) : 100;
    const std::string visual_path = "/results/euroc_vio_visual.csv";
    const std::string vio_path = "/results/euroc_stereo_vio.csv";
    const std::string command =
        "MY_SLAM_TRAJECTORY_PATH=" + visual_path +
        " /workspace/build/run_keyframe_vo " +
        std::to_string(start) + " " + std::to_string(end) + " 1 1";

    if (std::system(command.c_str()) != 0)
    {
        std::cerr << "Visual pose-only frontend failed.\n";
        return 1;
    }

    const auto visual = loadTrajectory(visual_path);
    if (visual.size() < 2)
    {
        std::cerr << "Insufficient visual poses for VIO.\n";
        return 1;
    }

    const std::filesystem::path dataset =
        "/datasets/EuRoC/MH_01_easy";
    my_slam::EurocReader reader(dataset);
    if (!reader.loadImu()) return 1;
    const auto& samples = reader.imu();
    const auto cam0 = my_slam::CameraModel::loadEuroc(
        dataset / "mav0" / "cam0" / "sensor.yaml"
    );
    const auto& imu_calibration = reader.imuCalibration();

    // T_BS maps sensor coordinates S to body coordinates B.
    // T_W_C maps camera coordinates C to world coordinates W.
    // Therefore T_W_C = T_W_B * T_BS_cam0 and
    // T_W_B = T_W_C * inverse(T_BS_cam0).
    const Eigen::Matrix4d T_BS_cam0 = cam0.T_BS();
    Eigen::Matrix4d T_W_B =
        visual.front().T_W_C * T_BS_cam0.inverse();
    Eigen::Quaterniond q_W_B(T_W_B.block<3, 3>(0, 0));
    Eigen::Vector3d p_W_B = T_W_B.block<3, 1>(0, 3);
    Eigen::Vector3d v_W_B = Eigen::Vector3d::Zero();
    Eigen::Vector3d initial_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_a = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_g = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity(0.0, 0.0, -9.81);

    const double initial_dt =
        (visual[1].timestamp - visual[0].timestamp) * 1e-9;
    if (initial_dt <= 0.0) return 1;
    v_W_B = (
        visual[1].T_W_C.block<3, 1>(0, 3) -
        visual[0].T_W_C.block<3, 1>(0, 3)
    ) / initial_dt;
    initial_velocity = v_W_B;

    if (!samples.empty())
    {
        Eigen::Vector3d mean_accel = Eigen::Vector3d::Zero();
        const std::size_t count = std::min<std::size_t>(20, samples.size());
        for (std::size_t i = 0; i < count; ++i)
            mean_accel += samples[i].accel;
        mean_accel /= static_cast<double>(count);
        const Eigen::Vector3d g_body = -mean_accel;
        if (g_body.norm() > 1e-6)
            gravity = q_W_B * (9.81 * g_body.normalized());
    }

    std::ofstream output(vio_path);
    output << "frame,timestamp_ns,x_m,y_m,z_m,qx,qy,qz,qw\n";
    std::size_t imu_index = 0;
    std::size_t intervals = 0;
    std::size_t missing = 0;
    std::size_t minimum = std::numeric_limits<std::size_t>::max();
    std::size_t maximum = 0;
    std::size_t total_interval_samples = 0;
    double disagreement_position_sum = 0.0;
    double disagreement_position_max = 0.0;
    double disagreement_rotation_sum = 0.0;
    double disagreement_rotation_max = 0.0;

    writePose(output, start, visual.front(), visual.front().T_W_C);
    for (std::size_t i = 1; i < visual.size(); ++i)
    {
        const auto t0 = visual[i - 1].timestamp;
        const auto t1 = visual[i].timestamp;
        std::size_t interval_count = 0;
        Eigen::Vector3d p = p_W_B;
        Eigen::Quaterniond q = q_W_B;
        Eigen::Vector3d v = v_W_B;
        my_slam::Timestamp previous_time = t0;

        while (imu_index < samples.size() && samples[imu_index].timestamp_ns <= t1)
        {
            if (samples[imu_index].timestamp_ns > t0)
            {
                const auto& sample = samples[imu_index];
                const double dt =
                    (sample.timestamp_ns - previous_time) * 1e-9;
                if (dt > 0.0 && dt < 0.1)
                {
                    const Eigen::Vector3d omega = sample.gyro - b_g;
                    const double angle = omega.norm() * dt;
                    if (angle > 1e-12)
                        q = q * Eigen::Quaterniond(
                            Eigen::AngleAxisd(angle, omega.normalized())
                        );
                    q.normalize();
                    const Eigen::Vector3d acceleration =
                        q * (sample.accel - b_a) + gravity;
                    p += v * dt + 0.5 * acceleration * dt * dt;
                    v += acceleration * dt;
                    previous_time = sample.timestamp_ns;
                    ++interval_count;
                }
            }
            ++imu_index;
        }

        if (previous_time < t1)
        {
            ++missing;
            std::cerr << "Missing IMU boundary coverage at interval "
                      << i << '\n';
        }
        ++intervals;
        total_interval_samples += interval_count;
        minimum = std::min(minimum, interval_count);
        maximum = std::max(maximum, interval_count);

        const Eigen::Matrix4d visual_T_W_B =
            visual[i].T_W_C * T_BS_cam0.inverse();
        const Eigen::Vector3d visual_position =
            visual_T_W_B.block<3, 1>(0, 3);
        const Eigen::Quaterniond visual_q(
            visual_T_W_B.block<3, 3>(0, 0)
        );
        const double position_error = (p - visual_position).norm();
        const double rotation_error = rotationDifferenceDeg(
            visual_q.toRotationMatrix(), q.toRotationMatrix()
        );
        disagreement_position_sum += position_error;
        disagreement_position_max = std::max(disagreement_position_max, position_error);
        disagreement_rotation_sum += rotation_error;
        disagreement_rotation_max = std::max(disagreement_rotation_max, rotation_error);

        // Explicit propagation/correction fusion: IMU affects both state
        // position and orientation, while visual pose remains the metric reference.
        p_W_B = 0.8 * visual_position + 0.2 * p;
        q_W_B = visual_q.slerp(0.2, q);
        q_W_B.normalize();
        v_W_B = 0.8 *
            ((visual_position - p_W_B) / std::max(initial_dt, 1e-6)) + 0.2 * v;
        T_W_B = Eigen::Matrix4d::Identity();
        T_W_B.block<3, 3>(0, 0) = q_W_B.toRotationMatrix();
        T_W_B.block<3, 1>(0, 3) = p_W_B;
        const Eigen::Matrix4d T_W_C = T_W_B * T_BS_cam0;
        if (!T_W_C.allFinite() || !validRotation(T_W_C.block<3, 3>(0, 0)))
        {
            std::cerr << "Invalid fused VIO state.\n";
            return 1;
        }
        writePose(output, i + start, visual[i], T_W_C);
        if (i == 1 || i % 20 == 0)
            std::cout << "VIO frame=" << i + start
                      << " visualPos=" << visual_position.transpose()
                      << " imuPos=" << p.transpose()
                      << " fusedPos=" << p_W_B.transpose()
                      << " visualImuRotDeg=" << rotation_error << '\n';
    }
    output.close();

    std::cout << "\n====================================\n"
              << "       Stereo VIO Summary\n"
              << "====================================\n"
              << "Visual frames: " << visual.size() << '\n'
              << "IMU samples: " << samples.size() << '\n'
              << "IMU intervals: " << intervals << '\n'
              << "Mean samples/interval: "
              << (intervals ? static_cast<double>(total_interval_samples) / intervals : 0.0)
              << '\n'
              << "Minimum samples/interval: " << minimum << '\n'
              << "Maximum samples/interval: " << maximum << '\n'
              << "Missing IMU intervals: " << missing << '\n'
              << "Initial velocity: " << initial_velocity.transpose() << '\n'
              << "Initial accel bias: 0 0 0\n"
              << "Initial gyro bias: 0 0 0\n"
              << "Gravity: " << gravity.transpose() << '\n'
              << "Mean visual/IMU position disagreement: "
              << disagreement_position_sum / std::max<std::size_t>(1, intervals)
              << " m\n"
              << "Max visual/IMU position disagreement: "
              << disagreement_position_max << " m\n"
              << "Mean visual/IMU rotation disagreement: "
              << disagreement_rotation_sum / std::max<std::size_t>(1, intervals)
              << " deg\n"
              << "Max visual/IMU rotation disagreement: "
              << disagreement_rotation_max << " deg\n"
              << "Bias optimization: disabled\n"
              << "Architecture: Stereo Visual-Inertial Odometry with "
                 "IMU propagation/fusion and visual correction\n"
              << "Trajectory:\n" << vio_path << '\n';
    return missing == 0 ? 0 : 1;
}
