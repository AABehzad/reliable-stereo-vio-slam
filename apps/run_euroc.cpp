#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/frontend/feature_matcher.hpp"
#include "my_slam/frontend/orb_extractor.hpp"
#include "my_slam/geometry/two_view_geometry.hpp"
#include "my_slam/io/euroc_reader.hpp"

int main(int argc, char** argv)
{
    const std::filesystem::path dataset_path =
        "/datasets/EuRoC/MH_01_easy";

    // By default use a small temporal separation.
    std::size_t frame_index_1 = 0;
    std::size_t frame_index_2 = 5;

    if (argc >= 3)
    {
        frame_index_1 =
            std::stoul(argv[1]);

        frame_index_2 =
            std::stoul(argv[2]);
    }

    std::cout
        << "=================================\n"
        << "      My SLAM - Two View VO\n"
        << "=================================\n\n";

    // --------------------------------------------------
    // Dataset
    // --------------------------------------------------

    my_slam::EurocReader reader(
        dataset_path
    );

    if (!reader.loadCam0())
    {
        std::cerr
            << "Failed to load cam0.\n";

        return 1;
    }

    const auto& records =
        reader.cam0();

    if (frame_index_1 >= records.size() ||
        frame_index_2 >= records.size())
    {
        std::cerr
            << "Frame index outside dataset range.\n";

        return 1;
    }

    if (frame_index_1 == frame_index_2)
    {
        std::cerr
            << "Two different frames are required.\n";

        return 1;
    }

    std::cout
        << "Frames loaded: "
        << records.size()
        << '\n';

    std::cout
        << "Using frames: "
        << frame_index_1
        << " -> "
        << frame_index_2
        << '\n';

    // --------------------------------------------------
    // Camera calibration
    // --------------------------------------------------

    const auto sensor_yaml =
        dataset_path
        / "mav0"
        / "cam0"
        / "sensor.yaml";

    my_slam::CameraModel camera;

    try
    {
        camera =
            my_slam::CameraModel::loadEuroc(
                sensor_yaml
            );
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "Calibration error: "
            << error.what()
            << '\n';

        return 1;
    }

    std::cout
        << "\nCamera calibration:\n";

    std::cout
        << "fx = "
        << camera.fx()
        << '\n';

    std::cout
        << "fy = "
        << camera.fy()
        << '\n';

    std::cout
        << "cx = "
        << camera.cx()
        << '\n';

    std::cout
        << "cy = "
        << camera.cy()
        << '\n';

    // --------------------------------------------------
    // Load images
    // --------------------------------------------------

    const auto& record1 =
        records[frame_index_1];

    const auto& record2 =
        records[frame_index_2];

    cv::Mat image1 =
        cv::imread(
            record1.image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    cv::Mat image2 =
        cv::imread(
            record2.image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    if (image1.empty() ||
        image2.empty())
    {
        std::cerr
            << "Failed to load input images.\n";

        return 1;
    }

    // --------------------------------------------------
    // ORB
    // --------------------------------------------------

    my_slam::OrbExtractor extractor(
        1200,
        1.2f,
        8
    );

    const auto features1 =
        extractor.extract(image1);

    const auto features2 =
        extractor.extract(image2);

    std::cout
        << "\nFrame "
        << frame_index_1
        << " keypoints: "
        << features1.keypoints.size()
        << '\n';

    std::cout
        << "Frame "
        << frame_index_2
        << " keypoints: "
        << features2.keypoints.size()
        << '\n';

    // --------------------------------------------------
    // Descriptor matching
    // --------------------------------------------------

    my_slam::FeatureMatcher matcher(
        0.75f
    );

    const auto match_result =
        matcher.match(
            features1,
            features2
        );

    std::cout
        << "\nRaw matches: "
        << match_result.raw_match_count
        << '\n';

    std::cout
        << "Ratio-test matches: "
        << match_result.good_matches.size()
        << '\n';

    // --------------------------------------------------
    // Save descriptor matches
    // --------------------------------------------------

    cv::Mat raw_match_image;

    cv::drawMatches(
        image1,
        features1.keypoints,
        image2,
        features2.keypoints,
        match_result.good_matches,
        raw_match_image,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    cv::imwrite(
        "/results/euroc_ratio_matches.png",
        raw_match_image
    );

    // --------------------------------------------------
    // Geometric verification
    // --------------------------------------------------

    my_slam::TwoViewGeometry geometry;

    const auto pose =
        geometry.estimate(
            features1,
            features2,
            match_result.good_matches,
            camera
        );

    if (!pose.success)
    {
        std::cerr
            << "\nRelative pose estimation failed.\n";

        return 1;
    }

    const double inlier_ratio =
        static_cast<double>(
            pose.inlier_matches.size()
        )
        /
        static_cast<double>(
            match_result.good_matches.size()
        );

    std::cout
        << "\nRANSAC / pose inliers: "
        << pose.inlier_matches.size()
        << '\n';

    std::cout
        << "Geometric inlier ratio: "
        << inlier_ratio * 100.0
        << " %\n";

    // --------------------------------------------------
    // Relative rotation
    // --------------------------------------------------

    std::cout
        << "\nRelative rotation R:\n"
        << pose.rotation
        << '\n';

    // --------------------------------------------------
    // Translation direction
    // --------------------------------------------------

    std::cout
        << "\nTranslation direction t:\n"
        << pose.translation
        << '\n';

    std::cout
        << "\nIMPORTANT:\n"
        << "Translation currently has NO metric scale.\n"
        << "Monocular two-view geometry only recovers "
        << "translation direction.\n";

    // --------------------------------------------------
    // Rotation magnitude
    // --------------------------------------------------

    cv::Mat rotation_vector;

    cv::Rodrigues(
        pose.rotation,
        rotation_vector
    );

    const double rotation_angle_rad =
        cv::norm(rotation_vector);

    const double rotation_angle_deg =
        rotation_angle_rad
        * 180.0
        / CV_PI;

    std::cout
        << "\nRelative rotation magnitude: "
        << rotation_angle_deg
        << " degrees\n";

    // --------------------------------------------------
    // Save RANSAC inliers
    // --------------------------------------------------

    cv::Mat inlier_image;

    cv::drawMatches(
        image1,
        features1.keypoints,
        image2,
        features2.keypoints,
        pose.inlier_matches,
        inlier_image,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    const std::filesystem::path output =
        "/results/euroc_ransac_inliers.png";

    if (!cv::imwrite(
            output.string(),
            inlier_image))
    {
        std::cerr
            << "Failed to save RANSAC result.\n";

        return 1;
    }

    std::cout
        << "\nRANSAC inlier visualization:\n"
        << output
        << '\n';

    std::cout
        << "\n=================================\n"
        << "Two-view pose estimation complete.\n"
        << "=================================\n";

    return 0;
}
