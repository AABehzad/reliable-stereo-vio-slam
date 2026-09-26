#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "my_slam/core/types.hpp"

namespace my_slam
{

struct ImageRecord
{
    Timestamp timestamp_ns = 0;
    std::filesystem::path image_path;
};

class EurocReader
{
public:
    explicit EurocReader(
        const std::filesystem::path& sequence_path
    );

    bool loadCam0();
    bool loadCam1();

    const std::vector<ImageRecord>& cam0() const;
    const std::vector<ImageRecord>& cam1() const;

private:
    bool loadCamera(
        const std::string& camera_name,
        std::vector<ImageRecord>& output
    );

    std::filesystem::path sequence_path_;

    std::vector<ImageRecord> cam0_records_;
    std::vector<ImageRecord> cam1_records_;
};

} // namespace my_slam
