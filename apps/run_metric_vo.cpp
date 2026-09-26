#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/imgcodecs.hpp>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/frontend/feature_matcher.hpp"
#include "my_slam/frontend/orb_extractor.hpp"
#include "my_slam/geometry/pnp_solver.hpp"
#include "my_slam/geometry/stereo_geometry.hpp"
#include "my_slam/io/euroc_reader.hpp"

namespace
{

Eigen::Matrix4d makeTransform(
    const cv::Mat& R,
    const cv::Mat& t)
{
    Eigen::Matrix4d T =
        Eigen::Matrix4d::Identity();

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            T(r, c) =
                R.at<double>(r, c);
        }

        T(r, 3) =
            t.at<double>(r, 0);
    }

    return T;
}

void writePose(
    std::ofstream& csv,
    std::size_t frame_index,
    my_slam::Timestamp timestamp,
    const Eigen::Matrix4d& T_W_C)
{
    const Eigen::Matrix3d R_W_C =
        T_W_C.block<3, 3>(0, 0);

    Eigen::Quaterniond q(R_W_C);
    q.normalize();

    csv
        << frame_index << ','
        << timestamp << ','
        << T_W_C(0, 3) << ','
        << T_W_C(1, 3) << ','
        << T_W_C(2, 3) << ','
        << q.x() << ','
        << q.y() << ','
        << q.z() << ','
        << q.w()
        << '\n';
}

} // namespace


int main(int argc, char** argv)
{
    const std::filesystem::path dataset =
        "/datasets/EuRoC/MH_01_easy";

    std::size_t start_frame = 0;
    std::size_t end_frame = 200;
    std::size_t stride = 1;

    if (argc >= 2)
    {
        start_frame =
            std::stoul(argv[1]);
    }

    if (argc >= 3)
    {
        end_frame =
            std::stoul(argv[2]);
    }

    if (argc >= 4)
    {
        stride =
            std::stoul(argv[3]);
    }

    if (stride == 0)
    {
        std::cerr
            << "Stride must be greater than zero.\n";

        return 1;
    }

    // --------------------------------------------------
    // Dataset
    // --------------------------------------------------

    my_slam::EurocReader reader(dataset);

    if (!reader.loadCam0() ||
        !reader.loadCam1())
    {
        std::cerr
            << "Failed to load EuRoC stereo data.\n";

        return 1;
    }

    const auto& cam0_records =
        reader.cam0();

    const auto& cam1_records =
        reader.cam1();

    const std::size_t frame_count =
        std::min(
            cam0_records.size(),
            cam1_records.size()
        );

    if (frame_count < 2)
    {
        std::cerr
            << "Dataset contains too few frames.\n";

        return 1;
    }

    if (start_frame >= frame_count)
    {
        std::cerr
            << "Start frame is outside dataset.\n";

        return 1;
    }

    end_frame =
        std::min(
            end_frame,
            frame_count - 1
        );

    if (end_frame <= start_frame)
    {
        std::cerr
            << "End frame must be greater than start frame.\n";

        return 1;
    }

    // --------------------------------------------------
    // Calibration
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
    // SLAM components
    // --------------------------------------------------

    my_slam::OrbExtractor extractor(
        1600,
        1.2f,
        8
    );

    my_slam::FeatureMatcher matcher(
        0.75f
    );

    my_slam::StereoGeometry stereo;
    my_slam::PnPSolver pnp;

    // --------------------------------------------------
    // World pose
    //
    // T_W_C = camera coordinates -> world coordinates.
    //
    // Initial camera frame becomes our temporary world.
    // --------------------------------------------------

    Eigen::Matrix4d T_W_C =
        Eigen::Matrix4d::Identity();

    // --------------------------------------------------
    // Output trajectory
    // --------------------------------------------------

    std::ofstream csv(
        "/results/euroc_metric_vo.csv"
    );

    if (!csv.is_open())
    {
        std::cerr
            << "Could not open trajectory output file.\n";

        return 1;
    }

    csv
        << "frame,timestamp_ns,"
        << "x_m,y_m,z_m,"
        << "qx,qy,qz,qw\n";

    csv << std::setprecision(15);

    writePose(
        csv,
        start_frame,
        cam0_records[start_frame].timestamp_ns,
        T_W_C
    );

    std::size_t successful_steps = 0;

    double total_translation = 0.0;

    std::cout
        << "====================================\n"
        << "       Metric Stereo VO\n"
        << "====================================\n";

    std::cout
        << "Start frame : "
        << start_frame
        << '\n';

    std::cout
        << "End frame   : "
        << end_frame
        << '\n';

    std::cout
        << "Stride      : "
        << stride
        << "\n\n";

    // --------------------------------------------------
    // VO loop
    // --------------------------------------------------

    for (std::size_t current = start_frame;
         current + stride <= end_frame;
         current += stride)
    {
        const std::size_t next =
            current + stride;

        // --------------------------------------------------
        // Verify stereo timestamp synchronization
        // --------------------------------------------------

        if (cam0_records[current].timestamp_ns !=
            cam1_records[current].timestamp_ns)
        {
            std::cerr
                << "Stereo synchronization failure at frame "
                << current
                << '\n';

            break;
        }

        // --------------------------------------------------
        // Load current stereo pair + next left image
        // --------------------------------------------------

        const cv::Mat left_current =
            cv::imread(
                cam0_records[current]
                    .image_path.string(),
                cv::IMREAD_GRAYSCALE
            );

        const cv::Mat right_current =
            cv::imread(
                cam1_records[current]
                    .image_path.string(),
                cv::IMREAD_GRAYSCALE
            );

        const cv::Mat left_next =
            cv::imread(
                cam0_records[next]
                    .image_path.string(),
                cv::IMREAD_GRAYSCALE
            );

        if (left_current.empty() ||
            right_current.empty() ||
            left_next.empty())
        {
            std::cerr
                << "Image loading failure at frame "
                << current
                << '\n';

            break;
        }

        // --------------------------------------------------
        // ORB
        // --------------------------------------------------

        const auto left_features =
            extractor.extract(
                left_current
            );

        const auto right_features =
            extractor.extract(
                right_current
            );

        const auto next_features =
            extractor.extract(
                left_next
            );

        // --------------------------------------------------
        // Stereo matching
        // --------------------------------------------------

        const auto stereo_matches =
            matcher.match(
                left_features,
                right_features
            );

        // --------------------------------------------------
        // Metric triangulation
        // --------------------------------------------------

        const auto stereo_result =
            stereo.triangulate(
                left_features,
                right_features,
                stereo_matches.good_matches,
                cam0,
                cam1
            );

        if (!stereo_result.success)
        {
            std::cerr
                << "Triangulation failure at frame "
                << current
                << '\n';

            break;
        }

        // --------------------------------------------------
        // Temporal matching
        // --------------------------------------------------

        const auto temporal_matches =
            matcher.match(
                left_features,
                next_features
            );

        // --------------------------------------------------
        // Map current left feature index -> metric 3D point
        // --------------------------------------------------

        std::unordered_map<int, cv::Point3f>
            landmark_lookup;

        landmark_lookup.reserve(
            stereo_result.landmarks.size()
        );

        for (const auto& landmark :
             stereo_result.landmarks)
        {
            landmark_lookup[
                landmark.match.queryIdx
            ] =
                cv::Point3f(
                    static_cast<float>(
                        landmark.point_cam0_m.x),
                    static_cast<float>(
                        landmark.point_cam0_m.y),
                    static_cast<float>(
                        landmark.point_cam0_m.z)
                );
        }

        // --------------------------------------------------
        // 3D -> 2D correspondences
        // --------------------------------------------------

        std::vector<cv::Point3f>
            object_points;

        std::vector<cv::Point2f>
            image_points;

        object_points.reserve(
            temporal_matches.good_matches.size()
        );

        image_points.reserve(
            temporal_matches.good_matches.size()
        );

        for (const auto& match :
             temporal_matches.good_matches)
        {
            const auto found =
                landmark_lookup.find(
                    match.queryIdx
                );

            if (found ==
                landmark_lookup.end())
            {
                continue;
            }

            object_points.push_back(
                found->second
            );

            image_points.push_back(
                next_features
                    .keypoints
                    .at(match.trainIdx)
                    .pt
            );
        }

        if (object_points.size() < 6)
        {
            std::cerr
                << "Too few 3D-2D correspondences at "
                << current
                << " -> "
                << next
                << '\n';

            break;
        }

        // --------------------------------------------------
        // PnP
        //
        // Returned transformation:
        //
        // T_Cnext_Ccurrent
        //
        // X_Cnext =
        // T_Cnext_Ccurrent * X_Ccurrent
        // --------------------------------------------------

        const auto pose =
            pnp.estimate(
                object_points,
                image_points,
                cam0
            );

        if (!pose.success)
        {
            std::cerr
                << "PnP failure at "
                << current
                << " -> "
                << next
                << '\n';

            break;
        }

        const Eigen::Matrix4d
            T_Cnext_Ccurrent =
                makeTransform(
                    pose.rotation_matrix,
                    pose.tvec
                );

        // --------------------------------------------------
        // To update camera pose in world:
        //
        // T_W_Cnext =
        // T_W_Ccurrent *
        // inverse(T_Cnext_Ccurrent)
        // --------------------------------------------------

        const Eigen::Matrix4d
            T_Ccurrent_Cnext =
                T_Cnext_Ccurrent.inverse();

        T_W_C =
            T_W_C
            * T_Ccurrent_Cnext;

        const double relative_translation =
            pose.tvec.empty()
            ? 0.0
            : cv::norm(pose.tvec);

        total_translation +=
            relative_translation;

        ++successful_steps;

        // --------------------------------------------------
        // Save pose
        // --------------------------------------------------

        writePose(
            csv,
            next,
            cam0_records[next].timestamp_ns,
            T_W_C
        );

        // --------------------------------------------------
        // Console progress
        // --------------------------------------------------

        if (successful_steps == 1 ||
            successful_steps % 20 == 0 ||
            next == end_frame)
        {
            std::cout
                << "Frame "
                << current
                << " -> "
                << next
                << " | landmarks="
                << stereo_result.landmarks.size()
                << " | 3D-2D="
                << object_points.size()
                << " | PnP inliers="
                << pose.pnp_inlier_count
                << " | position=["
                << T_W_C(0, 3)
                << ", "
                << T_W_C(1, 3)
                << ", "
                << T_W_C(2, 3)
                << "] m\n";
        }
    }

    csv.close();

    std::cout
        << "\n====================================\n"
        << "             VO Summary\n"
        << "====================================\n";

    std::cout
        << "Successful steps: "
        << successful_steps
        << '\n';

    std::cout
        << "Accumulated relative motion: "
        << total_translation
        << " m\n";

    std::cout
        << "\nFinal camera position:\n";

    std::cout
        << "X = "
        << T_W_C(0, 3)
        << " m\n";

    std::cout
        << "Y = "
        << T_W_C(1, 3)
        << " m\n";

    std::cout
        << "Z = "
        << T_W_C(2, 3)
        << " m\n";

    std::cout
        << "\nTrajectory saved to:\n"
        << "/results/euroc_metric_vo.csv\n";

    if (successful_steps == 0)
    {
        return 1;
    }

    std::cout
        << "\nMetric VO completed.\n";

    return 0;
}
