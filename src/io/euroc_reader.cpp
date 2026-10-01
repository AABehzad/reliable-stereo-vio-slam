#include "my_slam/io/euroc_reader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace my_slam
{

namespace
{

void trimEnd(std::string& value)
{
    while (!value.empty() &&
           (value.back() == '\r' ||
            value.back() == '\n' ||
            value.back() == ' '  ||
            value.back() == '\t'))
    {
        value.pop_back();
    }
}

} // namespace

EurocReader::EurocReader(
    const std::filesystem::path& sequence_path
)
    : sequence_path_(sequence_path)
{
}

bool EurocReader::loadCamera(
    const std::string& camera_name,
    std::vector<ImageRecord>& output
)
{
    output.clear();

    const auto camera_dir =
        sequence_path_ / "mav0" / camera_name;

    const auto csv_path =
        camera_dir / "data.csv";

    const auto image_dir =
        camera_dir / "data";

    if (!std::filesystem::exists(csv_path))
    {
        std::cerr
            << "CSV not found: "
            << csv_path << '\n';

        return false;
    }

    std::ifstream file(csv_path);

    if (!file.is_open())
    {
        return false;
    }

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::stringstream stream(line);

        std::string timestamp_string;
        std::string filename;

        if (!std::getline(
                stream,
                timestamp_string,
                ','))
        {
            continue;
        }

        if (!std::getline(
                stream,
                filename,
                ','))
        {
            continue;
        }

        trimEnd(timestamp_string);
        trimEnd(filename);

        try
        {
            const Timestamp timestamp =
                std::stoll(timestamp_string);

            const auto image_path =
                image_dir / filename;

            if (!std::filesystem::exists(image_path))
            {
                continue;
            }

            output.push_back(
                {
                    timestamp,
                    image_path
                }
            );
        }
        catch (...)
        {
            continue;
        }
    }

    std::cout
        << "Loaded "
        << camera_name
        << " frames: "
        << output.size()
        << '\n';

    return !output.empty();
}

bool EurocReader::loadCam0()
{
    return loadCamera(
        "cam0",
        cam0_records_
    );
}

bool EurocReader::loadCam1()
{
    return loadCamera(
        "cam1",
        cam1_records_
    );
}


bool EurocReader::loadImu()
{
    imu_samples_.clear();
    const auto imu_dir = sequence_path_ / "mav0" / "imu0";
    const auto csv_path = imu_dir / "data.csv";
    const auto yaml_path = imu_dir / "sensor.yaml";

    if (!std::filesystem::exists(csv_path) ||
        !std::filesystem::exists(yaml_path))
    {
        std::cerr << "IMU data or calibration not found.\n";
        return false;
    }

    try
    {
        const YAML::Node config = YAML::LoadFile(yaml_path.string());
        if (config["rate_hz"])
            imu_calibration_.rate_hz = config["rate_hz"].as<double>();
        if (config["gyroscope_noise_density"])
            imu_calibration_.gyro_noise_density =
                config["gyroscope_noise_density"].as<double>();
        if (config["gyroscope_random_walk"])
            imu_calibration_.gyro_random_walk =
                config["gyroscope_random_walk"].as<double>();
        if (config["accelerometer_noise_density"])
            imu_calibration_.accel_noise_density =
                config["accelerometer_noise_density"].as<double>();
        if (config["accelerometer_random_walk"])
            imu_calibration_.accel_random_walk =
                config["accelerometer_random_walk"].as<double>();
        if (config["T_BS"] && config["T_BS"]["data"])
        {
            const auto data = config["T_BS"]["data"].as<std::vector<double>>();
            if (data.size() != 16) return false;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    imu_calibration_.T_BS(r, c) = data[r * 4 + c];
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "Failed to parse IMU calibration: "
                  << error.what() << '\n';
        return false;
    }

    std::ifstream file(csv_path);
    if (!file.is_open()) return false;
    std::string line;
    Timestamp previous = 0;
    bool have_previous = false;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream stream(line);
        std::string field;
        std::vector<double> values;
        while (std::getline(stream, field, ','))
            values.push_back(std::stod(field));
        if (values.size() < 7) continue;
        const Timestamp timestamp = static_cast<Timestamp>(values[0]);
        if (have_previous && timestamp <= previous)
        {
            std::cerr << "Non-monotonic IMU timestamp.\n";
            return false;
        }
        previous = timestamp;
        have_previous = true;
        imu_samples_.push_back({
            timestamp,
            Eigen::Vector3d(values[1], values[2], values[3]),
            Eigen::Vector3d(values[4], values[5], values[6])
        });
    }

    if (imu_samples_.empty()) return false;
    std::cout << "IMU samples loaded: " << imu_samples_.size()
              << " first=" << imu_samples_.front().timestamp_ns
              << " last=" << imu_samples_.back().timestamp_ns
              << " nominal_rate=" << imu_calibration_.rate_hz << " Hz\n";
    return true;
}

const std::vector<ImageRecord>&
EurocReader::cam0() const
{
    return cam0_records_;
}

const std::vector<ImageRecord>&
EurocReader::cam1() const
{
    return cam1_records_;
}

const std::vector<ImuSample>& EurocReader::imu() const
{
    return imu_samples_;
}

const ImuCalibration& EurocReader::imuCalibration() const
{
    return imu_calibration_;
}

} // namespace my_slam
