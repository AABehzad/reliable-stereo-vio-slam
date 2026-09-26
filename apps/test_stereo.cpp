#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "my_slam/io/euroc_reader.hpp"

int main()
{
    const std::filesystem::path dataset_path =
        "/datasets/EuRoC/MH_01_easy";

    my_slam::EurocReader reader(dataset_path);

    if (!reader.loadCam0() ||
        !reader.loadCam1())
    {
        std::cerr
            << "Failed to load stereo cameras.\n";

        return 1;
    }

    const auto& left_records =
        reader.cam0();

    const auto& right_records =
        reader.cam1();

    std::cout
        << "\n=============================\n"
        << "     Stereo Input Test\n"
        << "=============================\n";

    std::cout
        << "\ncam0 frames: "
        << left_records.size()
        << '\n';

    std::cout
        << "cam1 frames: "
        << right_records.size()
        << '\n';

    if (left_records.empty() ||
        right_records.empty())
    {
        return 1;
    }

    const auto& left_record =
        left_records.front();

    const auto& right_record =
        right_records.front();

    std::cout
        << "\ncam0 timestamp: "
        << left_record.timestamp_ns
        << '\n';

    std::cout
        << "cam1 timestamp: "
        << right_record.timestamp_ns
        << '\n';

    const auto time_difference =
        left_record.timestamp_ns
        - right_record.timestamp_ns;

    std::cout
        << "Timestamp difference [ns]: "
        << time_difference
        << '\n';

    cv::Mat left =
        cv::imread(
            left_record.image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    cv::Mat right =
        cv::imread(
            right_record.image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    if (left.empty() || right.empty())
    {
        std::cerr
            << "Failed to load stereo pair.\n";

        return 1;
    }

    cv::imwrite(
        "/results/euroc_stereo_left.png",
        left
    );

    cv::imwrite(
        "/results/euroc_stereo_right.png",
        right
    );

    std::cout
        << "\nLeft resolution: "
        << left.cols
        << " x "
        << left.rows
        << '\n';

    std::cout
        << "Right resolution: "
        << right.cols
        << " x "
        << right.rows
        << '\n';

    std::cout
        << "\nStereo input OK.\n";

    return 0;
}
