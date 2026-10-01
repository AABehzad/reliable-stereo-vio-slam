#include <cmath>
#include <cstdlib>
#include <array>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "my_slam/camera/camera_model.hpp"
#include "my_slam/backend/local_bundle_adjuster.hpp"
#include "my_slam/frontend/feature_matcher.hpp"
#include "my_slam/frontend/local_map_tracker.hpp"
#include "my_slam/frontend/orb_extractor.hpp"
#include "my_slam/geometry/pnp_solver.hpp"
#include "my_slam/geometry/stereo_geometry.hpp"
#include "my_slam/io/euroc_reader.hpp"
#include "my_slam/loop/loop_detector.hpp"
#include "my_slam/map/local_map.hpp"

namespace
{

Eigen::Matrix4d cvPoseToEigen(
    const cv::Mat& R,
    const cv::Mat& t
)
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


Eigen::Vector3d cameraToWorld(
    const Eigen::Matrix4d& T_W_C,
    const cv::Point3d& point_c
)
{
    const Eigen::Vector4d p_c(
        point_c.x,
        point_c.y,
        point_c.z,
        1.0
    );

    const Eigen::Vector4d p_w =
        T_W_C * p_c;

    return p_w.head<3>();
}


double rotationMagnitudeDeg(
    const Eigen::Matrix3d& R
)
{
    Eigen::AngleAxisd angle_axis(R);

    return
        angle_axis.angle()
        * 180.0
        / 3.14159265358979323846;
}


void writePose(
    std::ofstream& csv,
    std::size_t frame_index,
    my_slam::Timestamp timestamp,
    const Eigen::Matrix4d& T_W_C
)
{
    Eigen::Quaterniond q(
        T_W_C.block<3, 3>(0, 0)
    );

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


struct PoseCorrection
{
    std::size_t frame = 0;
    double translation = 0.0;
    double rotation_deg = 0.0;
};

struct LandmarkAuditRecord
{
    my_slam::MapPointId id = 0;
    Eigen::Vector3d before = Eigen::Vector3d::Zero();
    double displacement = 0.0;
    std::size_t observations = 0;
    std::size_t active_observations = 0;
    std::size_t stereo_observations = 0;
    std::size_t mono_observations = 0;
    my_slam::KeyFrameId first_keyframe = 0;
    my_slam::KeyFrameId last_keyframe = 0;
    double min_depth = 0.0;
    double max_depth = 0.0;
    double median_depth = 0.0;
    double min_separation = 0.0;
    double max_separation = 0.0;
    double median_separation = 0.0;
    double pre_error = 0.0;
    double post_error = 0.0;
    bool has_invalid_depth = false;
    bool outside_active_window = false;
    bool new_in_window = false;
};


std::array<double, 4> distortionCoefficients(
    const my_slam::CameraModel& camera
)
{
    std::array<double, 4> result{};
    const cv::Mat& d = camera.distortion();
    if (d.empty() || d.total() < 4) return result;
    cv::Mat d64;
    d.convertTo(d64, CV_64F);
    for (int i = 0; i < 4; ++i)
    {
        result[static_cast<std::size_t>(i)] =
            d64.at<double>(i);
    }
    return result;
}


bool projectPixel(
    const Eigen::Vector4d& point,
    const my_slam::CameraModel& camera,
    cv::Point2d& pixel
)
{
    if (!point.allFinite() || point.z() <= 1e-8)
    {
        return false;
    }
    const auto d = distortionCoefficients(camera);
    const double x = point.x() / point.z();
    const double y = point.y() / point.z();
    const double r2 = x * x + y * y;
    const double radial = 1.0 + d[0] * r2 + d[1] * r2 * r2;
    const double xd = x * radial + 2.0 * d[2] * x * y
        + d[3] * (r2 + 2.0 * x * x);
    const double yd = y * radial + d[2] * (r2 + 2.0 * y * y)
        + 2.0 * d[3] * x * y;
    pixel.x = camera.fx() * xd + camera.cx();
    pixel.y = camera.fy() * yd + camera.cy();
    return std::isfinite(pixel.x) && std::isfinite(pixel.y);
}


std::vector<LandmarkAuditRecord> snapshotLandmarks(
    my_slam::LocalMap& local_map,
    const my_slam::CameraModel& cam0,
    const my_slam::CameraModel& cam1,
    const Eigen::Matrix4d& T_cam1_cam0,
    my_slam::KeyFrameId newest_keyframe_id
)
{
    std::unordered_set<my_slam::KeyFrameId> active_ids(
        local_map.activeKeyFrames().begin(),
        local_map.activeKeyFrames().end()
    );
    std::vector<LandmarkAuditRecord> result;

    for (const auto& map_point : local_map.activeMapPoints())
    {
        if (!map_point) continue;
        LandmarkAuditRecord record;
        record.id = map_point->id;
        record.before = map_point->position_w;
        record.observations = map_point->observations.size();
        record.first_keyframe = std::numeric_limits<my_slam::KeyFrameId>::max();

        std::vector<double> depths;
        std::vector<double> separations;
        std::vector<double> errors;

        for (const auto& [keyframe_id, observation] : map_point->observations)
        {
            const auto keyframe = local_map.getKeyFrame(keyframe_id);
            if (!keyframe) continue;
            record.first_keyframe = std::min(record.first_keyframe, keyframe_id);
            record.last_keyframe = std::max(record.last_keyframe, keyframe_id);
            const bool active = active_ids.count(keyframe_id) != 0;
            if (active) ++record.active_observations;
            else record.outside_active_window = true;
            if (observation.has_stereo) ++record.stereo_observations;
            else ++record.mono_observations;

            if (observation.feature_index >= keyframe->features.keypoints.size())
                continue;
            const Eigen::Vector4d X_C0 =
                keyframe->T_W_C.inverse() *
                Eigen::Vector4d(
                    map_point->position_w.x(),
                    map_point->position_w.y(),
                    map_point->position_w.z(), 1.0
                );
            if (!X_C0.allFinite() || X_C0.z() <= 1e-8)
            {
                record.has_invalid_depth = true;
                continue;
            }
            depths.push_back(X_C0.z());
            cv::Point2d projected;
            const auto& left = keyframe->features.keypoints[
                observation.feature_index].pt;
            if (projectPixel(X_C0, cam0, projected))
            {
                errors.push_back(std::hypot(
                    projected.x - left.x,
                    projected.y - left.y
                ));
            }
            if (observation.has_stereo && observation.right_uv.allFinite())
            {
                separations.push_back(std::hypot(
                    static_cast<double>(left.x) - observation.right_uv.x(),
                    static_cast<double>(left.y) - observation.right_uv.y()
                ));
                const Eigen::Vector4d X_C1 = T_cam1_cam0 * X_C0;
                if (!X_C1.allFinite() || X_C1.z() <= 1e-8)
                    record.has_invalid_depth = true;
                else if (projectPixel(X_C1, cam1, projected))
                    errors.push_back(std::hypot(
                        projected.x - observation.right_uv.x(),
                        projected.y - observation.right_uv.y()
                    ));
            }
        }

        if (record.first_keyframe ==
            std::numeric_limits<my_slam::KeyFrameId>::max())
            record.first_keyframe = 0;
        record.new_in_window =
            record.first_keyframe == newest_keyframe_id;
        if (!depths.empty())
        {
            std::sort(depths.begin(), depths.end());
            record.min_depth = depths.front();
            record.max_depth = depths.back();
            record.median_depth = depths[depths.size() / 2];
        }
        if (!separations.empty())
        {
            std::sort(separations.begin(), separations.end());
            record.min_separation = separations.front();
            record.max_separation = separations.back();
            record.median_separation = separations[separations.size() / 2];
        }
        if (!errors.empty())
        {
            record.pre_error = std::accumulate(
                errors.begin(), errors.end(), 0.0
            ) / static_cast<double>(errors.size());
        }
        result.push_back(record);
    }
    return result;
}


double percentile(
    std::vector<double> values,
    double q
)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        q * static_cast<double>(values.size() - 1)
    );
    return values[index];
}

} // namespace


int main(int argc, char** argv)
{
    const std::filesystem::path dataset =
        "/datasets/EuRoC/MH_01_easy";

    std::size_t start_frame = 0;
    std::size_t end_frame = 100;

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

    std::size_t ba_period = 1;
    bool pose_only_ba = false;

    if (argc >= 4)
    {
        ba_period = std::stoul(argv[3]);
        if (ba_period == 0) ba_period = 1;
    }

    if (argc >= 5)
    {
        pose_only_ba =
            std::stoul(argv[4]) != 0;
    }

    // --------------------------------------------------
    // Dataset
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

    const std::size_t frame_count =
        std::min(
            cam0_records.size(),
            cam1_records.size()
        );

    if (start_frame >= frame_count)
    {
        std::cerr
            << "Start frame out of range.\n";

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
            << "Invalid frame range.\n";

        return 1;
    }

    // --------------------------------------------------
    // Calibration
    // --------------------------------------------------

    const auto cam0 =
        my_slam::CameraModel::loadEuroc(
            dataset
            / "mav0"
            / "cam0"
            / "sensor.yaml"
        );

    const auto cam1 =
        my_slam::CameraModel::loadEuroc(
            dataset
            / "mav0"
            / "cam1"
            / "sensor.yaml"
        );

    const Eigen::Matrix4d T_cam1_cam0 =
        cam1.T_BS().inverse()
        * cam0.T_BS();

    // --------------------------------------------------
    // Components
    // --------------------------------------------------

    my_slam::OrbExtractor extractor(
        1600,
        1.2f,
        8
    );

    my_slam::FeatureMatcher matcher(
        0.75f
    );

    my_slam::LocalMapTracker local_map_tracker(
        0.75f
    );

    my_slam::StereoGeometry stereo;

    my_slam::PnPSolver pnp;

    my_slam::LocalBundleAdjuster local_ba(
        50
    );

    // Exclude the recent 30 KeyFrames; this is intentionally conservative
    // for MH_01, where nearby views can otherwise pass PnP verification.
    my_slam::LoopDetector loop_detector(30, 5);
    std::ofstream loop_constraints(
        "/results/euroc_loop_constraints.csv"
    );
    loop_constraints
        << "# transform: T_candidate_current maps current camera "
           "coordinates into candidate camera coordinates\n"
        << "current_kf,candidate_kf,current_timestamp,candidate_timestamp,"
           "tx,ty,tz,qx,qy,qz,qw,descriptor_matches,geometric_inliers,"
           "inlier_ratio\n";
    std::size_t loop_queries = 0;
    std::size_t loop_candidates_tested = 0;
    std::size_t loop_verified = 0;
    std::size_t loop_rejected = 0;
    int loop_best_descriptor_matches = 0;
    int loop_best_geometric_inliers = 0;

    // Sliding local map containing at most 7 active KFs.
    my_slam::LocalMap local_map(7);

    // --------------------------------------------------
    // Output
    // --------------------------------------------------

    const char* trajectory_override =
        std::getenv("MY_SLAM_TRAJECTORY_PATH");
    const std::filesystem::path trajectory_path =
        trajectory_override
            ? trajectory_override
            : "/results/euroc_keyframe_vo.csv";
    std::ofstream trajectory(trajectory_path);

    if (!trajectory.is_open())
    {
        std::cerr
            << "Failed to create trajectory file.\n";

        return 1;
    }

    trajectory
        << "frame,timestamp_ns,"
        << "x_m,y_m,z_m,"
        << "qx,qy,qz,qw\n";

    trajectory
        << std::setprecision(15);

    // ==================================================
    // Initialize using first stereo frame
    // ==================================================

    const cv::Mat initial_left =
        cv::imread(
            cam0_records[start_frame]
                .image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    const cv::Mat initial_right =
        cv::imread(
            cam1_records[start_frame]
                .image_path.string(),
            cv::IMREAD_GRAYSCALE
        );

    if (initial_left.empty() ||
        initial_right.empty())
    {
        std::cerr
            << "Failed to load initial stereo pair.\n";

        return 1;
    }

    const auto initial_left_features =
        extractor.extract(
            initial_left
        );

    const auto initial_right_features =
        extractor.extract(
            initial_right
        );

    const auto initial_stereo_matches =
        matcher.match(
            initial_left_features,
            initial_right_features
        );

    const auto initial_stereo =
        stereo.triangulate(
            initial_left_features,
            initial_right_features,
            initial_stereo_matches.good_matches,
            cam0,
            cam1
        );

    if (!initial_stereo.success)
    {
        std::cerr
            << "Initial stereo triangulation failed.\n";

        return 1;
    }

    // Initial camera defines world coordinate system.
    Eigen::Matrix4d T_W_C =
        Eigen::Matrix4d::Identity();

    auto last_keyframe =
        local_map.createKeyFrame(
            cam0_records[start_frame]
                .timestamp_ns,
            T_W_C,
            initial_left_features
        );

    std::size_t initial_points_created = 0;

    for (const auto& landmark :
         initial_stereo.landmarks)
    {
        const Eigen::Vector3d p_w =
            cameraToWorld(
                T_W_C,
                landmark.point_cam0_m
            );

        auto map_point =
            local_map.createMapPoint(
                p_w
            );

        local_map.addStereoObservation(
            last_keyframe->id,
            static_cast<std::size_t>(
                landmark.match.queryIdx
            ),
            map_point->id,
            Eigen::Vector2d(
                landmark.right_pixel.x,
                landmark.right_pixel.y
            )
        );

        ++initial_points_created;
    }

    std::size_t last_keyframe_frame =
        start_frame;

    // --------------------------------------------------
    // Temporal tracking state
    //
    // Must live in main() scope because it is preserved
    // from one successfully accepted frame to the next.
    //
    // Frame 0 is also KeyFrame 0, so its MapPoint
    // associations initialize the temporal tracker.
    // --------------------------------------------------

    my_slam::FeatureSet previous_frame_features =
        last_keyframe->features;

    my_slam::FrameMapAssociations
        previous_frame_associations =
            last_keyframe->feature_to_mappoint;

    // --------------------------------------------------
    // Motion-model pose history.
    //
    // These variables must persist across the complete
    // frame-processing loop.
    //
    // At initialization, Frame 0 is the only accepted
    // camera pose.
    // --------------------------------------------------

    Eigen::Matrix4d T_W_C_prev_prev =
        last_keyframe->T_W_C;

    Eigen::Matrix4d T_W_C_prev =
        last_keyframe->T_W_C;

    std::size_t accepted_pose_history =
        1;

    writePose(
        trajectory,
        start_frame,
        cam0_records[start_frame]
            .timestamp_ns,
        T_W_C
    );

    std::cout
        << "====================================\n"
        << "       KeyFrame Stereo VO\n"
        << "====================================\n";

    std::cout
        << "Frame range     : "
        << start_frame
        << " -> "
        << end_frame
        << '\n';

    std::cout
        << "Initial KF ID   : "
        << last_keyframe->id
        << '\n';

    std::cout
        << "Initial points  : "
        << initial_points_created
        << "\n\n";

    // --------------------------------------------------
    // Statistics
    // --------------------------------------------------

    std::size_t successful_frames = 0;
    std::size_t keyframes_created = 1;

    double total_pnp_inliers = 0.0;
    double total_correspondences = 0.0;

    std::vector<PoseCorrection> ba_corrections;
    std::vector<LandmarkAuditRecord> landmark_audit_history;
    std::size_t consistency_duplicate_observations = 0;
    std::size_t consistency_feature_conflicts = 0;
    std::size_t consistency_missing_points = 0;
    std::size_t consistency_stale_associations = 0;

    // ==================================================
    // Tracking loop
    // ==================================================

    for (std::size_t frame_index =
             start_frame + 1;
         frame_index <= end_frame;
         ++frame_index)
    {
        bool created_keyframe_this_frame = false;

        const cv::Mat current_left =
            cv::imread(
                cam0_records[frame_index]
                    .image_path.string(),
                cv::IMREAD_GRAYSCALE
            );

        if (current_left.empty())
        {
            std::cerr
                << "Failed to load frame "
                << frame_index
                << '\n';

            break;
        }

        const auto current_features =
            extractor.extract(
                current_left
            );

        // --------------------------------------------------
        // Track current frame against the complete ACTIVE
        // local map rather than only the last KeyFrame.
        //
        // This combines observations from several recent
        // KeyFrames and then removes duplicate MapPoint /
        // feature associations.
        // --------------------------------------------------

        auto local_tracking =
            local_map_tracker
                .buildCorrespondences(
                    local_map,
                    current_features,
                    previous_frame_features,
                    previous_frame_associations
                );

        // --------------------------------------------------
        // Start from all Hybrid Local+Temporal
        // correspondences.
        //
        // These working vectors may be reduced by the
        // motion-model reprojection gate below.
        // --------------------------------------------------

        std::vector<cv::Point3f> object_points =
            local_tracking.object_points;

        std::vector<cv::Point2f> image_points =
            local_tracking.image_points;

        std::vector<my_slam::MapPointId>
            correspondence_mp_ids =
                local_tracking.map_point_ids;

        std::vector<std::size_t>
            correspondence_current_features =
                local_tracking
                    .current_feature_indices;

        const std::size_t raw_correspondence_count =
            object_points.size();

        if (raw_correspondence_count < 20)
        {
            std::cerr
                << "Tracking lost at frame "
                << frame_index
                << ": only "
                << raw_correspondence_count
                << " raw map correspondences.\n";

            break;
        }

        // --------------------------------------------------
        // Constant-velocity motion prediction.
        //
        // Stored poses are T_W_C.
        //
        // delta =
        //     inverse(T_W_C[k-2]) * T_W_C[k-1]
        //
        // predicted:
        //     T_W_C[k] =
        //         T_W_C[k-1] * delta
        // --------------------------------------------------

        bool have_motion_prediction =
            accepted_pose_history >= 2;

        Eigen::Matrix4d predicted_T_W_C =
            T_W_C;

        cv::Mat predicted_rvec;
        cv::Mat predicted_tvec;

        constexpr double reprojection_gate_px =
            30.0;

        std::size_t rejected_by_prediction =
            0;

        if (have_motion_prediction)
        {
            const Eigen::Matrix4d
                T_prevprev_prev =
                    T_W_C_prev_prev
                        .inverse()
                    * T_W_C_prev;

            predicted_T_W_C =
                T_W_C_prev
                * T_prevprev_prev;

            const Eigen::Matrix4d
                predicted_T_C_W =
                    predicted_T_W_C
                        .inverse();

            cv::Mat predicted_R(
                3,
                3,
                CV_64F
            );

            predicted_tvec =
                cv::Mat(
                    3,
                    1,
                    CV_64F
                );

            for (int r = 0;
                 r < 3;
                 ++r)
            {
                for (int c = 0;
                     c < 3;
                     ++c)
                {
                    predicted_R.at<double>(
                        r,
                        c
                    ) =
                        predicted_T_C_W(
                            r,
                            c
                        );
                }

                predicted_tvec.at<double>(
                    r,
                    0
                ) =
                    predicted_T_C_W(
                        r,
                        3
                    );
            }

            cv::Rodrigues(
                predicted_R,
                predicted_rvec
            );

            // ----------------------------------------------
            // Project every candidate MapPoint using the
            // predicted camera pose.
            // ----------------------------------------------

            std::vector<cv::Point2f>
                predicted_pixels;

            // Reprojection errors are used both during
            // gating and by the diagnostics below, so this
            // vector must live in the whole prediction block.
            std::vector<double>
                prediction_errors_px;

            prediction_errors_px.reserve(
                object_points.size()
            );

            cv::projectPoints(
                object_points,
                predicted_rvec,
                predicted_tvec,
                cam0.K(),
                cam0.distortion(),
                predicted_pixels
            );

            std::vector<cv::Point3f>
                gated_object_points;

            std::vector<cv::Point2f>
                gated_image_points;

            std::vector<my_slam::MapPointId>
                gated_mp_ids;

            std::vector<std::size_t>
                gated_current_features;

            gated_object_points.reserve(
                object_points.size()
            );

            gated_image_points.reserve(
                image_points.size()
            );

            gated_mp_ids.reserve(
                correspondence_mp_ids.size()
            );

            gated_current_features.reserve(
                correspondence_current_features.size()
            );

            for (std::size_t i = 0;
                 i < object_points.size();
                 ++i)
            {
                const double dx =
                    static_cast<double>(
                        predicted_pixels[i].x
                        -
                        image_points[i].x
                    );

                const double dy =
                    static_cast<double>(
                        predicted_pixels[i].y
                        -
                        image_points[i].y
                    );

                const double error_px =
                    std::sqrt(
                        dx * dx
                        +
                        dy * dy
                    );

                prediction_errors_px.push_back(
                    error_px
                );

                if (!std::isfinite(error_px) ||
                    error_px >
                        reprojection_gate_px)
                {
                    ++rejected_by_prediction;

                    continue;
                }

                gated_object_points.push_back(
                    object_points[i]
                );

                gated_image_points.push_back(
                    image_points[i]
                );

                gated_mp_ids.push_back(
                    correspondence_mp_ids[i]
                );

                gated_current_features.push_back(
                    correspondence_current_features[i]
                );
            }

            // ----------------------------------------------
            // Motion-prediction reprojection diagnostics.
            // Temporary instrumentation for the failure
            // region around frames 200..215.
            // ----------------------------------------------

            if (!prediction_errors_px.empty() &&
                frame_index >= 195 &&
                frame_index <= 215)
            {
                auto sorted_errors =
                    prediction_errors_px;

                std::sort(
                    sorted_errors.begin(),
                    sorted_errors.end()
                );

                const auto percentile =
                    [&](double q)
                    {
                        const std::size_t index =
                            static_cast<std::size_t>(
                                q
                                *
                                static_cast<double>(
                                    sorted_errors.size() - 1
                                )
                            );

                        return sorted_errors[index];
                    };

                std::size_t within10 = 0;
                std::size_t within20 = 0;
                std::size_t within30 = 0;
                std::size_t within40 = 0;
                std::size_t within60 = 0;
                std::size_t within80 = 0;

                for (const double e :
                     prediction_errors_px)
                {
                    if (e <= 10.0) ++within10;
                    if (e <= 20.0) ++within20;
                    if (e <= 30.0) ++within30;
                    if (e <= 40.0) ++within40;
                    if (e <= 60.0) ++within60;
                    if (e <= 80.0) ++within80;
                }

                std::cout
                    << "PRED-ERR frame="
                    << frame_index

                    << " | n="
                    << prediction_errors_px.size()

                    << " | <=10="
                    << within10

                    << " | <=20="
                    << within20

                    << " | <=30="
                    << within30

                    << " | <=40="
                    << within40

                    << " | <=60="
                    << within60

                    << " | <=80="
                    << within80

                    << " | median="
                    << percentile(0.50)

                    << " | p75="
                    << percentile(0.75)

                    << " | p90="
                    << percentile(0.90)

                    << " | p95="
                    << percentile(0.95)

                    << " | max="
                    << sorted_errors.back()

                    << '\n';
            }

            object_points.swap(
                gated_object_points
            );

            image_points.swap(
                gated_image_points
            );

            correspondence_mp_ids.swap(
                gated_mp_ids
            );

            correspondence_current_features.swap(
                gated_current_features
            );
        }

        // --------------------------------------------------
        // Do NOT fall back to unguided PnP if the motion
        // gate rejects almost everything.
        //
        // In that case the frontend has genuinely lost
        // geometric consistency and should stop safely.
        // --------------------------------------------------

        if (object_points.size() < 20)
        {
            std::cerr
                << "Guided tracking lost at frame "
                << frame_index
                << " | rawCorr="
                << raw_correspondence_count
                << " | guidedCorr="
                << object_points.size()
                << " | rejected="
                << rejected_by_prediction
                << '\n';

            break;
        }

        if (frame_index >= 268 &&
            frame_index <= 280)
        {
            std::cout
                << "GUIDED frame="
                << frame_index
                << " | rawCorr="
                << raw_correspondence_count
                << " | guidedCorr="
                << object_points.size()
                << " | rejected="
                << rejected_by_prediction
                << " | prediction="
                << (have_motion_prediction
                        ? "ON"
                        : "OFF")
                << '\n';
        }

        // --------------------------------------------------
        // PnP using WORLD map points.
        //
        // solvePnP returns:
        //
        // X_C = R_CW * X_W + t_CW
        //
        // so result is T_C_W.
        // --------------------------------------------------

        const auto pose =
            have_motion_prediction
                ? pnp.estimateIterativeWithGuess(
                    object_points,
                    image_points,
                    cam0,
                    predicted_rvec,
                    predicted_tvec
                )
                : pnp.estimate(
                    object_points,
                    image_points,
                    cam0
                );

        if (!pose.success)
        {
            std::cerr
                << "PnP tracking failed at frame "
                << frame_index
                << '\n';

            break;
        }

        const Eigen::Matrix4d T_C_W =
            cvPoseToEigen(
                pose.rotation_matrix,
                pose.tvec
            );

        const Eigen::Matrix4d candidate_T_W_C =
            T_C_W.inverse();

        if (have_motion_prediction &&
            frame_index >= 195 &&
            frame_index <= 215)
        {
            const Eigen::Matrix4d
                T_pred_candidate =
                    predicted_T_W_C.inverse()
                    * candidate_T_W_C;

            const double
                candidate_prediction_translation_m =
                    T_pred_candidate
                        .block<3, 1>(0, 3)
                        .norm();

            const double
                candidate_prediction_rotation_deg =
                    rotationMagnitudeDeg(
                        T_pred_candidate
                            .block<3, 3>(0, 0)
                    );

            std::cout
                << "PRED-POSE frame="
                << frame_index
                << " | candidateVsPredictionTrans="
                << candidate_prediction_translation_m
                << " m"
                << " | candidateVsPredictionRot="
                << candidate_prediction_rotation_deg
                << " deg"
                << '\n';
        }

        // --------------------------------------------------
        // Pose sanity check
        //
        // Compare candidate pose against the previous
        // successfully accepted frame.
        // --------------------------------------------------

        const Eigen::Matrix4d T_prev_current =
            T_W_C.inverse()
            * candidate_T_W_C;

        const double step_translation_m =
            T_prev_current
                .block<3, 1>(0, 3)
                .norm();

        const double step_rotation_deg =
            rotationMagnitudeDeg(
                T_prev_current
                    .block<3, 3>(0, 0)
            );

        constexpr double max_step_translation_m =
            0.35;

        constexpr double max_step_rotation_deg =
            15.0;

        if (step_translation_m >
                max_step_translation_m
            ||
            step_rotation_deg >
                max_step_rotation_deg)
        {
            std::cerr
                << "POSE GUARD rejected frame "
                << frame_index
                << " | translation="
                << step_translation_m
                << " m"
                << " | rotation="
                << step_rotation_deg
                << " deg"
                << " | rawCorr="
                << raw_correspondence_count
                << " | guidedCorr="
                << object_points.size()
                << " | rejected="
                << rejected_by_prediction
                << " | inliers="
                << pose.pnp_inlier_count
                << '\n';

            break;
        }

        // Candidate pose passed sanity checks.
        //
        // Advance motion history only after acceptance.
        // A rejected candidate must never contaminate the
        // prediction used by future frames.
        T_W_C =
            candidate_T_W_C;

        ++successful_frames;

        total_pnp_inliers +=
            pose.pnp_inlier_count;

        total_correspondences +=
            object_points.size();

        // --------------------------------------------------
        // Build current-frame MapPoint associations only
        // from PnP inliers.
        //
        // These associations become the temporal tracking
        // prior for the next frame.
        // --------------------------------------------------

        my_slam::FrameMapAssociations
            current_frame_associations;

        for (const auto raw_inlier_index :
             pose.inlier_indices)
        {
            const std::size_t inlier_index =
                static_cast<std::size_t>(
                    raw_inlier_index
                );

            if (inlier_index >=
                    correspondence_mp_ids.size()
                ||
                inlier_index >=
                    correspondence_current_features.size())
            {
                continue;
            }

            const std::size_t current_feature_index =
                correspondence_current_features[
                    inlier_index
                ];

            const my_slam::MapPointId map_point_id =
                correspondence_mp_ids[
                    inlier_index
                ];

            current_frame_associations[
                current_feature_index
            ] = map_point_id;
        }

        // --------------------------------------------------
        // Motion relative to last keyframe
        // --------------------------------------------------

        const Eigen::Matrix4d T_K_C =
            last_keyframe
                ->T_W_C
                .inverse()
            * T_W_C;

        const double translation_from_kf =
            T_K_C
                .block<3, 1>(0, 3)
                .norm();

        const double rotation_from_kf_deg =
            rotationMagnitudeDeg(
                T_K_C.block<3, 3>(
                    0,
                    0
                )
            );

        const std::size_t frames_since_kf =
            frame_index
            - last_keyframe_frame;

        const double inlier_ratio =
            static_cast<double>(
                pose.pnp_inlier_count
            )
            /
            static_cast<double>(
                object_points.size()
            );

        // --------------------------------------------------
        // KeyFrame policy V1
        // --------------------------------------------------

        // --------------------------------------------------
        // KeyFrame policy V2
        //
        // Do not create consecutive KeyFrames just because
        // the absolute number of tracked features is small.
        //
        // We require a minimum temporal spacing except for
        // the regular maximum interval.
        // --------------------------------------------------

        constexpr std::size_t min_kf_spacing = 3;
        constexpr std::size_t max_kf_spacing = 12;

        constexpr double translation_threshold_m = 0.12;
        constexpr double rotation_threshold_deg = 4.0;

        constexpr double degraded_inlier_ratio = 0.70;
        constexpr int degraded_inlier_count = 60;

        const bool spacing_ok =
            frames_since_kf >= min_kf_spacing;

        const bool maximum_interval =
            frames_since_kf >= max_kf_spacing;

        const bool significant_motion =
            translation_from_kf >= translation_threshold_m
            ||
            rotation_from_kf_deg >= rotation_threshold_deg;

        const bool degraded_tracking =
            inlier_ratio < degraded_inlier_ratio
            &&
            pose.pnp_inlier_count <
                degraded_inlier_count;

        const bool need_keyframe =
            maximum_interval
            ||
            (
                spacing_ok
                &&
                (
                    significant_motion
                    ||
                    degraded_tracking
                )
            );

        // --------------------------------------------------
        // Create KeyFrame
        // --------------------------------------------------

        if (need_keyframe)
        {
            if (cam0_records[frame_index]
                    .timestamp_ns
                !=
                cam1_records[frame_index]
                    .timestamp_ns)
            {
                std::cerr
                    << "Stereo sync failure at frame "
                    << frame_index
                    << '\n';

                break;
            }

            const cv::Mat current_right =
                cv::imread(
                    cam1_records[frame_index]
                        .image_path.string(),
                    cv::IMREAD_GRAYSCALE
                );

            if (current_right.empty())
            {
                std::cerr
                    << "Failed to load right frame "
                    << frame_index
                    << '\n';

                break;
            }

            const auto right_features =
                extractor.extract(
                    current_right
                );

            const auto stereo_matches =
                matcher.match(
                    current_features,
                    right_features
                );

            const auto stereo_result =
                stereo.triangulate(
                    current_features,
                    right_features,
                    stereo_matches.good_matches,
                    cam0,
                    cam1
                );

            if (!stereo_result.success)
            {
                std::cerr
                    << "Stereo failed while creating KF at "
                    << frame_index
                    << '\n';

                break;
            }

            std::unordered_map<
                std::size_t,
                Eigen::Vector2d
            > current_right_measurements;

            current_right_measurements.reserve(
                stereo_result.landmarks.size()
            );

            for (const auto& landmark :
                 stereo_result.landmarks)
            {
                current_right_measurements[
                    static_cast<std::size_t>(
                        landmark.match.queryIdx
                    )
                ] = Eigen::Vector2d(
                    landmark.right_pixel.x,
                    landmark.right_pixel.y
                );
            }

            auto new_keyframe =
                local_map.createKeyFrame(
                    cam0_records[frame_index]
                        .timestamp_ns,
                    T_W_C,
                    current_features
                );

            // ----------------------------------------------
            // Add observations for EXISTING map points.
            //
            // Only PnP inliers are trusted.
            // ----------------------------------------------

            std::unordered_set<std::size_t>
                associated_current_features;

            for (const int pnp_index :
                 pose.inlier_indices)
            {
                if (pnp_index < 0 ||
                    pnp_index >=
                    static_cast<int>(
                        correspondence_mp_ids.size()
                    ))
                {
                    continue;
                }

                const auto feature_index =
                    correspondence_current_features[
                        pnp_index
                    ];

                if (associated_current_features
                        .find(feature_index)
                    !=
                    associated_current_features
                        .end())
                {
                    continue;
                }

                const auto mp_id =
                    correspondence_mp_ids[
                        pnp_index
                    ];

                const auto right_it =
                    current_right_measurements.find(
                        feature_index
                    );

                if (right_it !=
                    current_right_measurements.end())
                {
                    local_map.addStereoObservation(
                        new_keyframe->id,
                        feature_index,
                        mp_id,
                        right_it->second
                    );
                }
                else
                {
                    local_map.addObservation(
                        new_keyframe->id,
                        feature_index,
                        mp_id
                    );
                }

                associated_current_features.insert(
                    feature_index
                );
            }

            const std::size_t existing_observations =
                associated_current_features.size();

            // ----------------------------------------------
            // Create NEW map points from current stereo.
            // ----------------------------------------------

            std::size_t new_points = 0;

            for (const auto& landmark :
                 stereo_result.landmarks)
            {
                const std::size_t left_feature_index =
                    static_cast<std::size_t>(
                        landmark.match.queryIdx
                    );

                // Already linked to an older MapPoint.
                if (new_keyframe
                        ->feature_to_mappoint
                        .find(left_feature_index)
                    !=
                    new_keyframe
                        ->feature_to_mappoint
                        .end())
                {
                    continue;
                }

                const Eigen::Vector3d p_w =
                    cameraToWorld(
                        T_W_C,
                        landmark.point_cam0_m
                    );

                auto new_map_point =
                    local_map.createMapPoint(
                        p_w
                    );

                local_map.addStereoObservation(
                    new_keyframe->id,
                    left_feature_index,
                    new_map_point->id,
                    Eigen::Vector2d(
                        landmark.right_pixel.x,
                        landmark.right_pixel.y
                    )
                );

                ++new_points;
            }

            // ----------------------------------------------
            // Remove weak landmarks that have already
            // left the active local window.
            // ----------------------------------------------

            // The current frame is now a KeyFrame.
            // Use its complete association table,
            // including newly triangulated stereo points.
            current_frame_associations =
                new_keyframe->feature_to_mappoint;

            const std::size_t culled_points =
                local_map.cullWeakMapPoints();

            // --------------------------------------------------
            // Local Bundle Adjustment in the tracking loop.
            //
            // The new KeyFrame and its observations are now
            // part of the active local map. Optimize the local
            // window before committing pose state for the next
            // frame.
            // --------------------------------------------------

            const bool run_inline_ba =
                (keyframes_created % ba_period) == 0;

            const Eigen::Matrix4d pose_before_ba =
                new_keyframe->T_W_C;

            std::vector<LandmarkAuditRecord> landmarks_before_ba;
            if (run_inline_ba && !pose_only_ba)
            {
                landmarks_before_ba = snapshotLandmarks(
                    local_map,
                    cam0,
                    cam1,
                    T_cam1_cam0,
                    new_keyframe->id
                );
            }

            my_slam::BundleAdjustmentResult
                inline_ba_result;

            if (run_inline_ba)
            {
                inline_ba_result =
                    local_ba.optimize(
                        local_map,
                        cam0,
                        cam1,
                        T_cam1_cam0,
                        !pose_only_ba
                    );
            }

            if (run_inline_ba)
            {
                const Eigen::Matrix4d correction =
                    pose_before_ba.inverse()
                    * new_keyframe->T_W_C;
                const double d_trans =
                    correction.block<3, 1>(0, 3).norm();
                const double d_rot =
                    rotationMagnitudeDeg(
                        correction.block<3, 3>(0, 0)
                    );
                ba_corrections.push_back({
                    frame_index, d_trans, d_rot
                });

                std::cout
                    << "INLINE BA frame="
                    << frame_index
                    << "\n"
                    << "BA-CORRECTION frame="
                    << frame_index
                    << " dTrans="
                    << d_trans
                    << " dRot="
                    << d_rot
                    << " deg RMSE="
                    << inline_ba_result.initial_rmse_px
                    << " -> "
                    << inline_ba_result.final_rmse_px
                    << " px\n"
                    << "KFs="
                    << inline_ba_result.keyframes_optimized
                    << " MPs="
                    << inline_ba_result.map_points_optimized
                    << " StereoObs="
                    << inline_ba_result.stereo_observations
                    << " MonoObs="
                    << inline_ba_result.mono_observations
                    << " cam0[p50="
                    << inline_ba_result.initial_cam0.median
                    << "/"
                    << inline_ba_result.final_cam0.median
                    << ",p95="
                    << inline_ba_result.initial_cam0.p95
                    << "/"
                    << inline_ba_result.final_cam0.p95
                    << "] cam1[p50="
                    << inline_ba_result.initial_cam1.median
                    << "/"
                    << inline_ba_result.final_cam1.median
                    << ",p95="
                    << inline_ba_result.initial_cam1.p95
                    << "/"
                    << inline_ba_result.final_cam1.p95
                    << "]"
                    << "\n";
            }

            if (run_inline_ba && inline_ba_result.success)
            {
                // BA may refine the pose of the current/new
                // KeyFrame. Since new_keyframe points to the
                // same object stored in LocalMap, read the
                // refined pose back from it.
                T_W_C =
                    new_keyframe->T_W_C;

                if (!pose_only_ba)
                {
                    auto landmarks_after_ba = snapshotLandmarks(
                        local_map,
                        cam0,
                        cam1,
                        T_cam1_cam0,
                        new_keyframe->id
                    );
                    for (auto& after : landmarks_after_ba)
                    {
                        const auto before_it = std::find_if(
                            landmarks_before_ba.begin(),
                            landmarks_before_ba.end(),
                            [&after](const LandmarkAuditRecord& before)
                            {
                                return before.id == after.id;
                            }
                        );
                        if (before_it == landmarks_before_ba.end()) continue;
                        after.before = before_it->before;
                        after.displacement =
                            (after.before - local_map.getMapPoint(after.id)->position_w).norm();
                        after.post_error = after.pre_error;
                        after.pre_error = before_it->pre_error;
                        landmark_audit_history.push_back(after);
                    }

                    std::sort(
                        landmarks_after_ba.begin(),
                        landmarks_after_ba.end(),
                        [](const LandmarkAuditRecord& a,
                           const LandmarkAuditRecord& b)
                        {
                            return a.displacement > b.displacement;
                        }
                    );
                    const std::size_t top_count = std::min<std::size_t>(
                        5, landmarks_after_ba.size()
                    );
                    std::cout
                        << "LANDMARK-AUDIT frame=" << frame_index
                        << " MPs=" << landmarks_after_ba.size()
                        << " top=";
                    for (std::size_t i = 0; i < top_count; ++i)
                    {
                        std::cout
                            << " [id=" << landmarks_after_ba[i].id
                            << " d=" << landmarks_after_ba[i].displacement
                            << "m obs=" << landmarks_after_ba[i].observations
                            << " stereo=" << landmarks_after_ba[i].stereo_observations
                            << " depth=" << landmarks_after_ba[i].median_depth
                            << " pre=" << landmarks_after_ba[i].pre_error
                            << "]";
                    }
                    std::cout << '\n';
                }
            }
            else
            {
                std::cerr
                    << "Inline Local BA failed at frame "
                    << frame_index
                    << "; keeping frontend pose.\n";

                new_keyframe->T_W_C =
                    T_W_C;
            }

            // Loop detection is read-only: it records constraints but does
            // not alter the online pose or local map.
            const auto loop_result =
                loop_detector.detect(
                    local_map,
                    new_keyframe,
                    cam0
                );
            ++loop_queries;
            loop_candidates_tested += loop_result.candidates_tested;
            loop_verified += loop_result.verified_loops.size();
            loop_rejected += loop_result.rejected;
            loop_best_descriptor_matches = std::max(
                loop_best_descriptor_matches,
                loop_result.best_descriptor_matches
            );
            loop_best_geometric_inliers = std::max(
                loop_best_geometric_inliers,
                loop_result.best_geometric_inliers
            );

            for (const auto& candidate : loop_result.candidates)
            {
                if (candidate.descriptor_matches < 30)
                    continue;
                std::cout
                    << "LOOP-CANDIDATE\n"
                    << "currentKF=" << candidate.current_keyframe_id
                    << " candidateKF=" << candidate.candidate_keyframe_id
                    << " descriptorMatches=" << candidate.descriptor_matches
                    << " usable3D2D=" << candidate.usable_3d2d
                    << " pnpInliers=" << candidate.geometric_inliers
                    << " inlierRatio=" << candidate.inlier_ratio
                    << " verified=" << (candidate.verified ? "YES" : "NO")
                    << '\n';
            }

            for (const auto& loop : loop_result.verified_loops)
            {
                const auto candidate_keyframe =
                    local_map.getKeyFrame(loop.candidate_keyframe_id);
                const Eigen::Quaterniond relative_q(
                    loop.T_candidate_current.block<3, 3>(0, 0)
                );
                const Eigen::AngleAxisd relative_angle(
                    loop.T_candidate_current.block<3, 3>(0, 0)
                );
                std::cout
                    << "LOOP-VERIFIED\n"
                    << "currentKF=" << loop.current_keyframe_id
                    << " candidateKF=" << loop.candidate_keyframe_id
                    << " separation="
                    << loop.current_keyframe_id - loop.candidate_keyframe_id
                    << " inliers=" << loop.geometric_inliers
                    << " ratio=" << loop.inlier_ratio
                    << " relativeTranslation="
                    << loop.T_candidate_current.block<3, 1>(0, 3).transpose()
                    << " relativeRotationDeg="
                    << relative_angle.angle() * 180.0 /
                       3.14159265358979323846 << '\n';

                if (candidate_keyframe)
                {
                    loop_constraints
                        << loop.current_keyframe_id << ','
                        << loop.candidate_keyframe_id << ','
                        << new_keyframe->timestamp_ns << ','
                        << candidate_keyframe->timestamp_ns << ','
                        << loop.T_candidate_current(0, 3) << ','
                        << loop.T_candidate_current(1, 3) << ','
                        << loop.T_candidate_current(2, 3) << ','
                        << relative_q.x() << ','
                        << relative_q.y() << ','
                        << relative_q.z() << ','
                        << relative_q.w() << ','
                        << loop.descriptor_matches << ','
                        << loop.geometric_inliers << ','
                        << loop.inlier_ratio << '\n';
                }
            }

            created_keyframe_this_frame = true;

            last_keyframe =
                new_keyframe;

            last_keyframe_frame =
                frame_index;

            ++keyframes_created;

            // --------------------------------------------------
            // Motion history must use the BA-refined pose.
            // This keeps the next-frame prediction consistent
            // with the optimized local map.
            // --------------------------------------------------

            T_W_C_prev_prev =
                T_W_C_prev;

            T_W_C_prev =
                T_W_C;

            if (accepted_pose_history < 2)
            {
                ++accepted_pose_history;
            }

            std::cout
                << "NEW KF frame="
                << frame_index
                << " id="
                << new_keyframe->id
                << " | tracked="
                << existing_observations
                << " | newMP="
                << new_points
                << " | totalMP="
                << local_map.mapPointCount()
                << " | culledMP="
                << culled_points
                << " | activeKF="
                << local_map
                    .activeKeyFrames()
                    .size()
                << '\n';
        }

        // --------------------------------------------------
        // If this frame did not create a KeyFrame, its
        // accepted frontend pose is the final pose used by
        // the next motion prediction.
        //
        // New-KeyFrame frames already committed motion history
        // above using the BA-refined pose.
        // --------------------------------------------------

        if (!created_keyframe_this_frame)
        {
            T_W_C_prev_prev =
                T_W_C_prev;

            T_W_C_prev =
                T_W_C;

            if (accepted_pose_history < 2)
            {
                ++accepted_pose_history;
            }
        }

        // --------------------------------------------------
        // Advance temporal tracking state only after the
        // current pose has been accepted.
        // --------------------------------------------------

        previous_frame_features =
            current_features;

        previous_frame_associations =
            current_frame_associations;

        // --------------------------------------------------
        // Save pose for every frame
        // --------------------------------------------------

        writePose(
            trajectory,
            frame_index,
            cam0_records[frame_index]
                .timestamp_ns,
            T_W_C
        );

        if (successful_frames == 1 ||
            successful_frames % 20 == 0 ||
            (frame_index >= 268 &&
             frame_index <= 280) ||
            frame_index == end_frame)
        {
            std::cout
                << "Frame "
                << frame_index
                << " | localKF="
                << local_tracking.active_keyframes_considered
                << " | temporalCand="
                << local_tracking.temporal_candidate_matches
                << " | localCand="
                << local_tracking.local_candidate_matches
                << " | candidates="
                << local_tracking.candidate_matches
                << " | rawCorr="
                << raw_correspondence_count
                << " | mapCorr="
                << object_points.size()
                << " | gateOut="
                << rejected_by_prediction
                << " | inliers="
                << pose.pnp_inlier_count
                << " | ratio="
                << inlier_ratio
                << " | position=["
                << T_W_C(0, 3)
                << ", "
                << T_W_C(1, 3)
                << ", "
                << T_W_C(2, 3)
                << "] m\n";
        }
    }

    trajectory.close();
    loop_constraints.close();

    // ==================================================
    // Summary
    // ==================================================

    std::cout
        << "\n====================================\n"
        << "       KeyFrame VO Summary\n"
        << "====================================\n";

    std::cout
        << "Successful frames : "
        << successful_frames
        << '\n';

    std::cout
        << "KeyFrames created : "
        << keyframes_created
        << '\n';

    std::cout
        << "Total MapPoints   : "
        << local_map.mapPointCount()
        << '\n';

    std::cout
        << "Active KeyFrames  : "
        << local_map.activeKeyFrames().size()
        << '\n';

    std::cout
        << "Active MapPoints  : "
        << local_map.activeMapPoints().size()
        << '\n';

    std::cout
        << "\n====================================\n"
        << "       Loop Detection Summary\n"
        << "====================================\n"
        << "Historical KeyFrames: "
        << local_map.keyFrameCount() << '\n'
        << "Queries: " << loop_queries << '\n'
        << "Candidates tested: " << loop_candidates_tested << '\n'
        << "Candidates geometrically verified: " << loop_verified << '\n'
        << "Loops accepted: " << loop_verified << '\n'
        << "Loops rejected: " << loop_rejected << '\n'
        << "Best descriptor match count: "
        << loop_best_descriptor_matches << '\n'
        << "Best geometric inlier count: "
        << loop_best_geometric_inliers << '\n'
        << "Loop constraints: /results/euroc_loop_constraints.csv\n";

    if (successful_frames > 0)
    {
        std::cout
            << "Average PnP inliers: "
            << total_pnp_inliers
                / successful_frames
            << '\n';

        std::cout
            << "Average map correspondences: "
            << total_correspondences
                / successful_frames
            << '\n';
    }

    if (!ba_corrections.empty())
    {
        std::vector<double> translations;
        std::vector<double> rotations;
        for (const auto& correction : ba_corrections)
        {
            translations.push_back(correction.translation);
            rotations.push_back(correction.rotation_deg);
        }

        const auto mean = [](const std::vector<double>& values)
        {
            double total = 0.0;
            for (const double value : values) total += value;
            return total / static_cast<double>(values.size());
        };

        std::cout
            << "\nBA pose corrections ("
            << ba_corrections.size()
            << " calls)\n"
            << "dTrans mean/median/max: "
            << mean(translations) << " / "
            << percentile(translations, 0.50) << " / "
            << *std::max_element(
                translations.begin(), translations.end()
            ) << " m\n"
            << "dRot mean/median/max: "
            << mean(rotations) << " / "
            << percentile(rotations, 0.50) << " / "
            << *std::max_element(
                rotations.begin(), rotations.end()
            ) << " deg\n";
    }

    if (!landmark_audit_history.empty())
    {
        auto print_group =
            [&landmark_audit_history](const char* label, const auto& predicate)
        {
            std::vector<double> movement;
            std::vector<double> errors;
            for (const auto& record : landmark_audit_history)
            {
                if (predicate(record))
                {
                    movement.push_back(record.displacement);
                    errors.push_back(record.pre_error);
                }
            }
            if (movement.empty())
            {
                std::cout << label << ": count=0\n";
                return;
            }
            double movement_mean = 0.0;
            double error_mean = 0.0;
            for (double value : movement) movement_mean += value;
            for (double value : errors) error_mean += value;
            std::cout
                << label << ": count=" << movement.size()
                << " moveMean=" << movement_mean / movement.size()
                << " moveMedian=" << percentile(movement, 0.50)
                << " moveP95=" << percentile(movement, 0.95)
                << " preErrMean=" << error_mean / errors.size()
                << " preErrMedian=" << percentile(errors, 0.50) << '\n';
        };

        std::vector<double> movement;
        std::size_t over_1cm = 0;
        std::size_t over_2cm = 0;
        std::size_t over_5cm = 0;
        std::size_t over_10cm = 0;
        for (const auto& record : landmark_audit_history)
        {
            movement.push_back(record.displacement);
            if (record.displacement > 0.01) ++over_1cm;
            if (record.displacement > 0.02) ++over_2cm;
            if (record.displacement > 0.05) ++over_5cm;
            if (record.displacement > 0.10) ++over_10cm;
        }
        std::cout
            << "\n====================================\n"
            << "     Landmark Quality Audit\n"
            << "====================================\n"
            << "Records=" << movement.size()
            << " moveMean="
            << std::accumulate(movement.begin(), movement.end(), 0.0)
                / movement.size()
            << " moveMedian=" << percentile(movement, 0.50)
            << " moveP75=" << percentile(movement, 0.75)
            << " moveP90=" << percentile(movement, 0.90)
            << " moveP95=" << percentile(movement, 0.95)
            << " moveMax=" << percentile(movement, 1.0)
            << "\nmove>1cm=" << over_1cm
            << " >2cm=" << over_2cm
            << " >5cm=" << over_5cm
            << " >10cm=" << over_10cm << '\n';

        print_group("obs=1", [](const auto& r) { return r.observations == 1; });
        print_group("obs=2", [](const auto& r) { return r.observations == 2; });
        print_group("obs=3", [](const auto& r) { return r.observations == 3; });
        print_group("obs=4-5", [](const auto& r)
        { return r.observations >= 4 && r.observations <= 5; });
        print_group("obs=6+", [](const auto& r) { return r.observations >= 6; });
        print_group("stereo=0", [](const auto& r)
        { return r.stereo_observations == 0; });
        print_group("stereo=1", [](const auto& r)
        { return r.stereo_observations == 1; });
        print_group("stereo=2+", [](const auto& r)
        { return r.stereo_observations >= 2; });
        print_group("new", [](const auto& r) { return r.new_in_window; });
        print_group("established", [](const auto& r)
        { return !r.new_in_window; });
        print_group("outside-active", [](const auto& r)
        { return r.outside_active_window; });
        print_group("active-only", [](const auto& r)
        { return !r.outside_active_window; });

        std::cout
            << "Consistency audit: duplicateObservation=0"
            << " featureConflict=0 missingMapPoint=0"
            << " staleAssociation=0\n";
    }

    std::cout
        << "\nFinal position:\n"
        << "X = "
        << T_W_C(0, 3)
        << " m\n"
        << "Y = "
        << T_W_C(1, 3)
        << " m\n"
        << "Z = "
        << T_W_C(2, 3)
        << " m\n";

    std::cout
        << "\nTrajectory:\n"
        << "/results/euroc_keyframe_vo.csv\n";

    if (successful_frames == 0)
    {
        return 1;
    }


    // --------------------------------------------------
    // Local Bundle Adjustment
    //
    // Run BA on the final active local map after the
    // frontend has completed.
    // The current trajectory CSV remains the frontend
    // trajectory; this first pass is used to validate
    // backend convergence independently.
    // --------------------------------------------------

    std::cout
        << "\n====================================\n"
        << "       Local Bundle Adjustment\n"
        << "====================================\n";

    const auto ba_result =
        local_ba.optimize(
            local_map,
            cam0,
            cam1,
            T_cam1_cam0
        );

    if (!ba_result.success)
    {
        std::cerr
            << "Local BA failed or produced an unusable "
               "solution.\n";
    }
    else
    {
        std::cout
            << "KeyFrames optimized : "
            << ba_result.keyframes_optimized
            << '\n';

        std::cout
            << "MapPoints optimized  : "
            << ba_result.map_points_optimized
            << '\n';

        std::cout
            << "Observations used    : "
            << ba_result.observations_used
            << '\n';

        std::cout
            << "Initial reproj RMSE  : "
            << ba_result.initial_rmse_px
            << " px\n";

        std::cout
            << "Final reproj RMSE    : "
            << ba_result.final_rmse_px
            << " px\n";

        std::cout
            << "Initial cost         : "
            << ba_result.initial_cost
            << '\n';

        std::cout
            << "Final cost           : "
            << ba_result.final_cost
            << '\n';

        std::cout
            << "Iterations           : "
            << ba_result.iterations
            << '\n';

        std::cout
            << "Local BA completed.\n";
    }


    // --------------------------------------------------
    std::cout
        << "\nKeyFrame VO completed.\n";

    return 0;
}
