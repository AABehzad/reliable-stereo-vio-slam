#include "my_slam/geometry/pnp_solver.hpp"

#include <cmath>

#include <opencv2/calib3d.hpp>

namespace my_slam
{

namespace
{

MetricPoseResult estimateImpl(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const CameraModel& camera,
    const cv::Mat* initial_rvec,
    const cv::Mat* initial_tvec,
    bool use_ransac
)
{
    MetricPoseResult result;

    if (object_points.size() < 6 ||
        image_points.size() < 6 ||
        object_points.size() != image_points.size())
    {
        return result;
    }

    bool use_initial_guess = false;

    if (initial_rvec != nullptr &&
        initial_tvec != nullptr &&
        !initial_rvec->empty() &&
        !initial_tvec->empty())
    {
        result.rvec =
            initial_rvec->clone();

        result.tvec =
            initial_tvec->clone();

        use_initial_guess = true;
    }

    cv::Mat inliers;

    bool ok = false;

    if (use_ransac)
    {
        ok =
            cv::solvePnPRansac(
                object_points,
                image_points,
                camera.K(),
                camera.distortion(),
                result.rvec,
                result.tvec,
                use_initial_guess,
                100,
                4.0f,
                0.99,
                inliers,
                cv::SOLVEPNP_ITERATIVE
            );
    }
    else
    {
        ok =
            cv::solvePnP(
                object_points,
                image_points,
                camera.K(),
                camera.distortion(),
                result.rvec,
                result.tvec,
                use_initial_guess,
                cv::SOLVEPNP_ITERATIVE
            );

        if (ok)
        {
            std::vector<cv::Point2f>
                projected;

            cv::projectPoints(
                object_points,
                result.rvec,
                result.tvec,
                camera.K(),
                camera.distortion(),
                projected
            );

            for (std::size_t i = 0;
                 i < projected.size();
                 ++i)
            {
                const double dx =
                    static_cast<double>(
                        projected[i].x
                        -
                        image_points[i].x
                    );

                const double dy =
                    static_cast<double>(
                        projected[i].y
                        -
                        image_points[i].y
                    );

                const double error =
                    std::sqrt(
                        dx * dx +
                        dy * dy
                    );

                if (std::isfinite(error) &&
                    error <= 4.0)
                {
                    inliers.push_back(
                        static_cast<int>(i)
                    );
                }
            }
        }
    }

    if (!ok)
    {
        return result;
    }

    cv::Rodrigues(
        result.rvec,
        result.rotation_matrix
    );

    if (!inliers.empty())
    {
        result.pnp_inlier_count =
            inliers.rows;

        result.inlier_indices.reserve(
            static_cast<std::size_t>(
                inliers.rows
            )
        );

        for (int i = 0;
             i < inliers.rows;
             ++i)
        {
            result.inlier_indices.push_back(
                inliers.at<int>(i, 0)
            );
        }
    }

    result.success =
        result.pnp_inlier_count >= 6;

    return result;
}

} // namespace


MetricPoseResult
PnPSolver::estimate(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const CameraModel& camera
) const
{
    return estimateImpl(
        object_points,
        image_points,
        camera,
        nullptr,
        nullptr,
        true
    );
}


MetricPoseResult
PnPSolver::estimateWithGuess(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const CameraModel& camera,
    const cv::Mat& initial_rvec,
    const cv::Mat& initial_tvec
) const
{
    return estimateImpl(
        object_points,
        image_points,
        camera,
        &initial_rvec,
        &initial_tvec,
        true
    );
}


MetricPoseResult
PnPSolver::estimateIterativeWithGuess(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const CameraModel& camera,
    const cv::Mat& initial_rvec,
    const cv::Mat& initial_tvec
) const
{
    return estimateImpl(
        object_points,
        image_points,
        camera,
        &initial_rvec,
        &initial_tvec,
        false
    );
}

} // namespace my_slam
