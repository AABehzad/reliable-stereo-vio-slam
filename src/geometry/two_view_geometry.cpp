#include "my_slam/geometry/two_view_geometry.hpp"

#include <opencv2/calib3d.hpp>

namespace my_slam
{

RelativePoseResult TwoViewGeometry::estimate(
    const FeatureSet& first,
    const FeatureSet& second,
    const std::vector<cv::DMatch>& matches,
    const CameraModel& camera
) const
{
    RelativePoseResult result;

    if (matches.size() < 8)
    {
        return result;
    }

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    points1.reserve(matches.size());
    points2.reserve(matches.size());

    for (const auto& match : matches)
    {
        points1.push_back(
            first.keypoints.at(match.queryIdx).pt
        );

        points2.push_back(
            second.keypoints.at(match.trainIdx).pt
        );
    }

    // -----------------------------------------
    // Undistort matched points
    // -----------------------------------------

    std::vector<cv::Point2f> undistorted1;
    std::vector<cv::Point2f> undistorted2;

    cv::undistortPoints(
        points1,
        undistorted1,
        camera.K(),
        camera.distortion(),
        cv::noArray(),
        camera.K()
    );

    cv::undistortPoints(
        points2,
        undistorted2,
        camera.K(),
        camera.distortion(),
        cv::noArray(),
        camera.K()
    );

    // -----------------------------------------
    // Essential Matrix + RANSAC
    // -----------------------------------------

    cv::Mat inlier_mask;

    result.essential_matrix =
        cv::findEssentialMat(
            undistorted1,
            undistorted2,
            camera.K(),
            cv::RANSAC,
            0.999,
            1.0,
            inlier_mask
        );

    if (result.essential_matrix.empty())
    {
        return result;
    }

    // -----------------------------------------
    // Recover relative camera pose
    // -----------------------------------------

    const int recovered_inliers =
        cv::recoverPose(
            result.essential_matrix,
            undistorted1,
            undistorted2,
            camera.K(),
            result.rotation,
            result.translation,
            inlier_mask
        );

    if (recovered_inliers < 8)
    {
        return result;
    }

    // -----------------------------------------
    // Collect geometrically valid matches
    // -----------------------------------------

    for (std::size_t i = 0; i < matches.size(); ++i)
    {
        if (inlier_mask.at<unsigned char>(
                static_cast<int>(i)) != 0)
        {
            result.inlier_matches.push_back(
                matches[i]
            );
        }
    }

    result.success =
        !result.inlier_matches.empty();

    return result;
}

} // namespace my_slam
