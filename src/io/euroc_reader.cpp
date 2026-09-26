#include "my_slam/io/euroc_reader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

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

} // namespace my_slam
