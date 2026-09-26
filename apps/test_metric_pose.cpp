#include <cmath>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/frontend/feature_matcher.hpp"
#include "my_slam/frontend/orb_extractor.hpp"
#include "my_slam/geometry/pnp_solver.hpp"
#include "my_slam/geometry/stereo_geometry.hpp"
#include "my_slam/io/euroc_reader.hpp"

int main(int argc, char** argv)
{
    const std::filesystem::path dataset =
        "/datasets/EuRoC/MH_01_easy";

    std::size_t frame_t = 0;
    std::size_t frame_t1 = 5;

    if (argc >= 3)
    {
        frame_t  = std::stoul(argv[1]);
        frame_t1 = std::stoul(argv[2]);
    }

    my_slam::EurocReader reader(dataset);

    if (!reader.loadCam0() ||
        !reader.loadCam1())
    {
        std::cerr
            << "Failed to load EuRoC stereo data.\n";
        return 1;
    }

    const auto& cam0_records = reader.cam0();
    const auto& cam1_records = reader.cam1();

    if (frame_t >= cam0_records.size() ||
        frame_t1 >= cam0_records.size() ||
        frame_t >= cam1_records.size() ||
        frame_t1 >= cam1_records.size())
    {
        std::cerr
            << "Frame indices out of range.\n";
        return 1;
    }

    // --------------------------------------------------
    // Load camera models
    // --------------------------------------------------

    const auto cam0 =
        my_slam::CameraModel::loadEuroc(
            dataset / "mav0" / "cam0" / "sensor.yaml"
        );

    const auto cam1 =
        my_slam::CameraModel::loadEuroc(
            dataset / "mav0" / "cam1" / "sensor.yaml"
        );

    // --------------------------------------------------
    // Load images:
    // left_t, right_t, left_t1
    // --------------------------------------------------

    const cv::Mat left_t =
        cv::imread(
            cam0_records[frame_t].image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    const cv::Mat right_t =
        cv::imread(
            cam1_records[frame_t].image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    const cv::Mat left_t1 =
        cv::imread(
            cam0_records[frame_t1].image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    if (left_t.empty() ||
        right_t.empty() ||
        left_t1.empty())
    {
        std::cerr
            << "Failed to load one or more images.\n";
        return 1;
    }

    // --------------------------------------------------
    // Feature extraction
    // --------------------------------------------------

    my_slam::OrbExtractor extractor(
        1600,
        1.2f,
        8
    );

    const auto left_features_t =
        extractor.extract(left_t);

    const auto right_features_t =
        extractor.extract(right_t);

    const auto left_features_t1 =
        extractor.extract(left_t1);

    std::cout
        << "Left_t features     : "
        << left_features_t.keypoints.size()
        << '\n';

    std::cout
        << "Right_t features    : "
        << right_features_t.keypoints.size()
        << '\n';

    std::cout
        << "Left_t+1 features   : "
        << left_features_t1.keypoints.size()
        << '\n';

    // --------------------------------------------------
    // Stereo matching at time t
    // --------------------------------------------------

    my_slam::FeatureMatcher matcher(0.75f);

    const auto stereo_matches =
        matcher.match(
            left_features_t,
            right_features_t
        );

    std::cout
        << "\nStereo ratio matches: "
        << stereo_matches.good_matches.size()
        << '\n';

    my_slam::StereoGeometry stereo;

    const auto stereo_result =
        stereo.triangulate(
            left_features_t,
            right_features_t,
            stereo_matches.good_matches,
            cam0,
            cam1
        );

    if (!stereo_result.success)
    {
        std::cerr
            << "Stereo triangulation failed.\n";
        return 1;
    }

    std::cout
        << "Triangulated landmarks: "
        << stereo_result.landmarks.size()
        << '\n';

    // --------------------------------------------------
    // Temporal matching: left_t -> left_t1
    // --------------------------------------------------

    const auto temporal_matches =
        matcher.match(
            left_features_t,
            left_features_t1
        );

    std::cout
        << "Temporal ratio matches: "
        << temporal_matches.good_matches.size()
        << '\n';

    // --------------------------------------------------
    // Build lookup:
    // left_t feature index -> 3D point
    // --------------------------------------------------

    std::unordered_map<int, cv::Point3f> landmark_lookup;

    landmark_lookup.reserve(
        stereo_result.landmarks.size()
    );

    for (const auto& landmark :
         stereo_result.landmarks)
    {
        const int left_query_index =
            landmark.match.queryIdx;

        landmark_lookup[left_query_index] =
            cv::Point3f(
                static_cast<float>(landmark.point_cam0_m.x),
                static_cast<float>(landmark.point_cam0_m.y),
                static_cast<float>(landmark.point_cam0_m.z)
            );
    }

    // --------------------------------------------------
    // Build 3D-2D correspondences
    // --------------------------------------------------

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::DMatch> visualization_matches;

    object_points.reserve(
        temporal_matches.good_matches.size()
    );

    image_points.reserve(
        temporal_matches.good_matches.size()
    );

    visualization_matches.reserve(
        temporal_matches.good_matches.size()
    );

    for (const auto& match :
         temporal_matches.good_matches)
    {
        const int left_t_index =
            match.queryIdx;

        auto it =
            landmark_lookup.find(left_t_index);

        if (it == landmark_lookup.end())
        {
            continue;
        }

        object_points.push_back(
            it->second
        );

        image_points.push_back(
            left_features_t1
                .keypoints
                .at(match.trainIdx)
                .pt
        );

        visualization_matches.push_back(match);
    }

    std::cout
        << "\n3D-2D correspondences: "
        << object_points.size()
        << '\n';

    if (object_points.size() < 6)
    {
        std::cerr
            << "Not enough 3D-2D correspondences for PnP.\n";
        return 1;
    }

    // --------------------------------------------------
    // PnP
    // --------------------------------------------------

    my_slam::PnPSolver pnp;

    const auto pose =
        pnp.estimate(
            object_points,
            image_points,
            cam0
        );

    if (!pose.success)
    {
        std::cerr
            << "PnP failed.\n";
        return 1;
    }

    std::cout
        << "\n=================================\n"
        << "        Metric PnP Pose\n"
        << "=================================\n";

    std::cout
        << "\nPnP inliers: "
        << pose.pnp_inlier_count
        << '\n';

    std::cout
        << "\nRotation matrix R:\n"
        << pose.rotation_matrix
        << '\n';

    std::cout
        << "\nTranslation t [meters]:\n"
        << pose.tvec
        << '\n';

    const double translation_norm =
        cv::norm(pose.tvec);

    std::cout
        << "\nTranslation magnitude: "
        << translation_norm
        << " m\n";

    cv::Mat rotation_vector;
    cv::Rodrigues(
        pose.rotation_matrix,
        rotation_vector
    );

    const double rotation_angle_rad =
        cv::norm(rotation_vector);

    const double rotation_angle_deg =
        rotation_angle_rad * 180.0 / CV_PI;

    std::cout
        << "Rotation magnitude: "
        << rotation_angle_deg
        << " deg\n";

    // --------------------------------------------------
    // Save inlier visualization
    // --------------------------------------------------

    std::vector<cv::DMatch> inlier_matches;

    inlier_matches.reserve(
        pose.inlier_indices.size()
    );

    for (const int idx : pose.inlier_indices)
    {
        if (idx >= 0 &&
            idx < static_cast<int>(
                visualization_matches.size()))
        {
            inlier_matches.push_back(
                visualization_matches[idx]
            );
        }
    }

    cv::Mat vis;

    cv::drawMatches(
        left_t,
        left_features_t.keypoints,
        left_t1,
        left_features_t1.keypoints,
        inlier_matches,
        vis,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    cv::imwrite(
        "/results/euroc_metric_pnp_inliers.png",
        vis
    );

    std::cout
        << "\nVisualization:\n"
        << "/results/euroc_metric_pnp_inliers.png\n";

    std::cout
        << "\nMetric PnP OK.\n";

    return 0;
}
