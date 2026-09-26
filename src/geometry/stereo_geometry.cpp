#include "my_slam/geometry/stereo_geometry.hpp"

#include <cmath>

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>

namespace my_slam
{

namespace
{

cv::Mat eigenRotationToCv(
    const Eigen::Matrix3d& R
)
{
    cv::Mat result(3, 3, CV_64F);

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            result.at<double>(r, c) = R(r, c);
        }
    }

    return result;
}

cv::Mat eigenTranslationToCv(
    const Eigen::Vector3d& t
)
{
    cv::Mat result(3, 1, CV_64F);

    for (int r = 0; r < 3; ++r)
    {
        result.at<double>(r, 0) = t(r);
    }

    return result;
}

double pixelDistance(
    const cv::Point2f& a,
    const cv::Point2f& b
)
{
    const double dx =
        static_cast<double>(a.x - b.x);

    const double dy =
        static_cast<double>(a.y - b.y);

    return std::sqrt(
        dx * dx + dy * dy
    );
}

} // namespace


StereoTriangulationResult
StereoGeometry::triangulate(
    const FeatureSet& left_features,
    const FeatureSet& right_features,
    const std::vector<cv::DMatch>& matches,
    const CameraModel& left_camera,
    const CameraModel& right_camera
) const
{
    StereoTriangulationResult result;

    result.input_match_count =
        matches.size();

    if (matches.size() < 8)
    {
        return result;
    }

    // --------------------------------------------------
    // Relative transformation:
    //
    // T_BS0 = Body <- Cam0
    // T_BS1 = Body <- Cam1
    //
    // Therefore:
    //
    // T_C1_C0 =
    // inverse(T_BS1) * T_BS0
    //
    // Cam0 point -> Cam1 point
    // --------------------------------------------------

    result.T_cam1_cam0 =
        right_camera.T_BS().inverse()
        * left_camera.T_BS();

    const Eigen::Matrix3d R10 =
        result.T_cam1_cam0
            .block<3, 3>(0, 0);

    const Eigen::Vector3d t10 =
        result.T_cam1_cam0
            .block<3, 1>(0, 3);

    result.baseline_m =
        t10.norm();

    // --------------------------------------------------
    // Collect matched pixels
    // --------------------------------------------------

    std::vector<cv::Point2f> left_points;
    std::vector<cv::Point2f> right_points;

    left_points.reserve(matches.size());
    right_points.reserve(matches.size());

    for (const auto& match : matches)
    {
        left_points.push_back(
            left_features
                .keypoints
                .at(match.queryIdx)
                .pt
        );

        right_points.push_back(
            right_features
                .keypoints
                .at(match.trainIdx)
                .pt
        );
    }

    // --------------------------------------------------
    // Undistort, but keep coordinates in pixel space
    // --------------------------------------------------

    std::vector<cv::Point2f> left_undistorted;
    std::vector<cv::Point2f> right_undistorted;

    cv::undistortPoints(
        left_points,
        left_undistorted,
        left_camera.K(),
        left_camera.distortion(),
        cv::noArray(),
        left_camera.K()
    );

    cv::undistortPoints(
        right_points,
        right_undistorted,
        right_camera.K(),
        right_camera.distortion(),
        cv::noArray(),
        right_camera.K()
    );

    // --------------------------------------------------
    // Geometrically verify stereo correspondences.
    // --------------------------------------------------

    cv::Mat fundamental_mask;

    const cv::Mat fundamental =
        cv::findFundamentalMat(
            left_undistorted,
            right_undistorted,
            cv::FM_RANSAC,
            1.0,
            0.999,
            fundamental_mask
        );

    if (fundamental.empty())
    {
        return result;
    }

    std::vector<cv::Point2f> left_inliers;
    std::vector<cv::Point2f> right_inliers;

    std::vector<cv::DMatch> inlier_matches;

    for (std::size_t i = 0;
         i < matches.size();
         ++i)
    {
        if (fundamental_mask.at<unsigned char>(
                static_cast<int>(i)) == 0)
        {
            continue;
        }

        left_inliers.push_back(
            left_undistorted[i]
        );

        right_inliers.push_back(
            right_undistorted[i]
        );

        inlier_matches.push_back(
            matches[i]
        );
    }

    result.epipolar_inlier_count =
        inlier_matches.size();

    if (inlier_matches.size() < 8)
    {
        return result;
    }

    // --------------------------------------------------
    // Projection matrices
    //
    // P0 = K0 [I | 0]
    // P1 = K1 [R10 | t10]
    // --------------------------------------------------

    cv::Mat Rt0 =
        cv::Mat::zeros(
            3,
            4,
            CV_64F
        );

    const cv::Mat identity3 =
        cv::Mat::eye(
            3,
            3,
            CV_64F
        );

    identity3.copyTo(
        Rt0(
            cv::Rect(0, 0, 3, 3)
        )
    );

    cv::Mat Rt1 =
        cv::Mat::zeros(
            3,
            4,
            CV_64F
        );

    const cv::Mat R10_cv =
        eigenRotationToCv(R10);

    const cv::Mat t10_cv =
        eigenTranslationToCv(t10);

    R10_cv.copyTo(
        Rt1(
            cv::Rect(0, 0, 3, 3)
        )
    );

    t10_cv.copyTo(
        Rt1(
            cv::Rect(3, 0, 1, 3)
        )
    );

    const cv::Mat P0 =
        left_camera.K() * Rt0;

    const cv::Mat P1 =
        right_camera.K() * Rt1;

    // --------------------------------------------------
    // Triangulation
    // --------------------------------------------------

    cv::Mat homogeneous_points;

    cv::triangulatePoints(
        P0,
        P1,
        left_inliers,
        right_inliers,
        homogeneous_points
    );

    homogeneous_points.convertTo(
        homogeneous_points,
        CV_64F
    );

    // --------------------------------------------------
    // Validate 3D landmarks
    // --------------------------------------------------

    constexpr double max_reprojection_error_px =
        2.5;

    for (int i = 0;
         i < homogeneous_points.cols;
         ++i)
    {
        const double w =
            homogeneous_points.at<double>(3, i);

        if (std::abs(w) < 1e-12)
        {
            continue;
        }

        Eigen::Vector3d point_cam0(
            homogeneous_points.at<double>(0, i) / w,
            homogeneous_points.at<double>(1, i) / w,
            homogeneous_points.at<double>(2, i) / w
        );

        // Point must be in front of cam0.
        if (point_cam0.z() <= 0.0)
        {
            continue;
        }

        const Eigen::Vector3d point_cam1 =
            R10 * point_cam0 + t10;

        // Point must also be in front of cam1.
        if (point_cam1.z() <= 0.0)
        {
            continue;
        }

        // --------------------------------------------------
        // Reprojection in cam0
        // --------------------------------------------------

        cv::Point2f projected_left;

        projected_left.x =
            static_cast<float>(
                left_camera.fx()
                * point_cam0.x()
                / point_cam0.z()
                + left_camera.cx()
            );

        projected_left.y =
            static_cast<float>(
                left_camera.fy()
                * point_cam0.y()
                / point_cam0.z()
                + left_camera.cy()
            );

        // --------------------------------------------------
        // Reprojection in cam1
        // --------------------------------------------------

        cv::Point2f projected_right;

        projected_right.x =
            static_cast<float>(
                right_camera.fx()
                * point_cam1.x()
                / point_cam1.z()
                + right_camera.cx()
            );

        projected_right.y =
            static_cast<float>(
                right_camera.fy()
                * point_cam1.y()
                / point_cam1.z()
                + right_camera.cy()
            );

        const double left_error =
            pixelDistance(
                projected_left,
                left_inliers[i]
            );

        const double right_error =
            pixelDistance(
                projected_right,
                right_inliers[i]
            );

        const double reprojection_error =
            0.5
            * (left_error + right_error);

        if (reprojection_error >
            max_reprojection_error_px)
        {
            continue;
        }

        StereoLandmark landmark;

        landmark.left_pixel =
            left_inliers[i];

        landmark.right_pixel =
            right_inliers[i];

        landmark.point_cam0_m =
            cv::Point3d(
                point_cam0.x(),
                point_cam0.y(),
                point_cam0.z()
            );

        landmark.match =
            inlier_matches[i];

        landmark.reprojection_error =
            reprojection_error;

        result.landmarks.push_back(
            landmark
        );
    }

    result.success =
        !result.landmarks.empty();

    return result;
}

} // namespace my_slam
