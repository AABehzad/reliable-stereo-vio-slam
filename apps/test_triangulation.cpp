#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/frontend/feature_matcher.hpp"
#include "my_slam/frontend/orb_extractor.hpp"
#include "my_slam/geometry/stereo_geometry.hpp"
#include "my_slam/io/euroc_reader.hpp"

int main()
{
    const std::filesystem::path dataset =
        "/datasets/EuRoC/MH_01_easy";

    // --------------------------------------------------
    // Load stereo dataset
    // --------------------------------------------------

    my_slam::EurocReader reader(dataset);

    if (!reader.loadCam0() ||
        !reader.loadCam1())
    {
        std::cerr
            << "Failed to load stereo dataset.\n";

        return 1;
    }

    const auto& cam0_records =
        reader.cam0();

    const auto& cam1_records =
        reader.cam1();

    if (cam0_records.empty() ||
        cam1_records.empty())
    {
        return 1;
    }

    if (cam0_records.front().timestamp_ns !=
        cam1_records.front().timestamp_ns)
    {
        std::cerr
            << "First stereo pair is not synchronized.\n";

        return 1;
    }

    // --------------------------------------------------
    // Calibration
    // --------------------------------------------------

    const auto cam0_yaml =
        dataset
        / "mav0"
        / "cam0"
        / "sensor.yaml";

    const auto cam1_yaml =
        dataset
        / "mav0"
        / "cam1"
        / "sensor.yaml";

    const auto cam0 =
        my_slam::CameraModel::loadEuroc(
            cam0_yaml
        );

    const auto cam1 =
        my_slam::CameraModel::loadEuroc(
            cam1_yaml
        );

    // --------------------------------------------------
    // Images
    // --------------------------------------------------

    const cv::Mat left =
        cv::imread(
            cam0_records.front()
                .image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    const cv::Mat right =
        cv::imread(
            cam1_records.front()
                .image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    if (left.empty() || right.empty())
    {
        std::cerr
            << "Failed to load stereo images.\n";

        return 1;
    }

    // --------------------------------------------------
    // Features
    // --------------------------------------------------

    my_slam::OrbExtractor extractor(
        1600,
        1.2f,
        8
    );

    const auto left_features =
        extractor.extract(left);

    const auto right_features =
        extractor.extract(right);

    std::cout
        << "Left features : "
        << left_features.keypoints.size()
        << '\n';

    std::cout
        << "Right features: "
        << right_features.keypoints.size()
        << '\n';

    // --------------------------------------------------
    // Descriptor matching
    // --------------------------------------------------

    my_slam::FeatureMatcher matcher(
        0.75f
    );

    const auto matches =
        matcher.match(
            left_features,
            right_features
        );

    std::cout
        << "\nRaw descriptor queries: "
        << matches.raw_match_count
        << '\n';

    std::cout
        << "Ratio-test matches: "
        << matches.good_matches.size()
        << '\n';

    // --------------------------------------------------
    // Stereo triangulation
    // --------------------------------------------------

    my_slam::StereoGeometry stereo;

    const auto result =
        stereo.triangulate(
            left_features,
            right_features,
            matches.good_matches,
            cam0,
            cam1
        );

    std::cout
        << "\n=================================\n"
        << "       Stereo Geometry\n"
        << "=================================\n";

    std::cout
        << "\nT_cam1_cam0:\n"
        << result.T_cam1_cam0
        << '\n';

    std::cout
        << "\nStereo baseline: "
        << result.baseline_m
        << " m\n";

    std::cout
        << "\nInput matches: "
        << result.input_match_count
        << '\n';

    std::cout
        << "Epipolar inliers: "
        << result.epipolar_inlier_count
        << '\n';

    std::cout
        << "Valid metric landmarks: "
        << result.landmarks.size()
        << '\n';

    if (!result.success)
    {
        std::cerr
            << "\nStereo triangulation failed.\n";

        return 1;
    }

    // --------------------------------------------------
    // Display first few 3D landmarks
    // --------------------------------------------------

    std::cout
        << "\nFirst 10 metric landmarks "
        << "[cam0 coordinates, meters]:\n\n";

    const std::size_t count =
        std::min<std::size_t>(
            10,
            result.landmarks.size()
        );

    for (std::size_t i = 0;
         i < count;
         ++i)
    {
        const auto& point =
            result.landmarks[i]
                .point_cam0_m;

        std::cout
            << "[" << i << "] "
            << "X = " << point.x
            << "  Y = " << point.y
            << "  Z = " << point.z
            << " m\n";
    }

    // --------------------------------------------------
    // Save valid matches visualization
    // --------------------------------------------------

    std::vector<cv::DMatch> valid_matches;

    valid_matches.reserve(
        result.landmarks.size()
    );

    for (const auto& landmark :
         result.landmarks)
    {
        valid_matches.push_back(
            landmark.match
        );
    }

    cv::Mat visualization;

    cv::drawMatches(
        left,
        left_features.keypoints,
        right,
        right_features.keypoints,
        valid_matches,
        visualization,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    cv::imwrite(
        "/results/euroc_stereo_triangulated.png",
        visualization
    );

    // --------------------------------------------------
    // Save 3D landmarks CSV
    // --------------------------------------------------

    std::ofstream csv(
        "/results/euroc_landmarks_cam0.csv"
    );

    csv
        << "x_m,y_m,z_m,reprojection_error_px\n";

    csv << std::setprecision(12);

    for (const auto& landmark :
         result.landmarks)
    {
        csv
            << landmark.point_cam0_m.x
            << ','
            << landmark.point_cam0_m.y
            << ','
            << landmark.point_cam0_m.z
            << ','
            << landmark.reprojection_error
            << '\n';
    }

    std::cout
        << "\nVisualization:\n"
        << "/results/euroc_stereo_triangulated.png\n";

    std::cout
        << "\n3D landmarks:\n"
        << "/results/euroc_landmarks_cam0.csv\n";

    std::cout
        << "\nStereo triangulation OK.\n";

    return 0;
}
