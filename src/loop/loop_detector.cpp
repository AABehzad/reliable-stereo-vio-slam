#include "my_slam/loop/loop_detector.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <Eigen/Geometry>
#include <opencv2/features2d.hpp>

namespace my_slam
{

namespace
{

double relativeRotationDegrees(const Eigen::Matrix4d& transform)
{
    const Eigen::AngleAxisd angle_axis(
        transform.block<3, 3>(0, 0)
    );
    return angle_axis.angle() * 180.0 /
        3.14159265358979323846;
}

} // namespace

LoopDetector::LoopDetector(
    std::size_t minimum_separation,
    std::size_t shortlist_size
)
    : minimum_separation_(minimum_separation),
      shortlist_size_(shortlist_size)
{
}


LoopDetectionResult LoopDetector::detect(
    LocalMap& local_map,
    const std::shared_ptr<KeyFrame>& current,
    const CameraModel& camera
) const
{
    LoopDetectionResult result;
    if (!current || current->features.descriptors.empty())
        return result;

    result.queries = 1;
    result.historical_keyframes =
        current->id >= minimum_separation_
            ? current->id - minimum_separation_
            : 0;

    struct Ranked
    {
        std::shared_ptr<KeyFrame> keyframe;
        std::vector<cv::DMatch> matches;
    };
    std::vector<Ranked> ranked;
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);

    for (KeyFrameId id = 0;
         id < current->id;
         ++id)
    {
        if (current->id - id < minimum_separation_)
            continue;
        const auto candidate = local_map.getKeyFrame(id);
        if (!candidate || candidate->features.descriptors.empty())
            continue;

        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(
            current->features.descriptors,
            candidate->features.descriptors,
            knn,
            2
        );
        std::vector<cv::DMatch> matches;
        for (const auto& pair : knn)
        {
            if (pair.size() < 2) continue;
            if (pair[0].distance < 0.75f * pair[1].distance)
                matches.push_back(pair[0]);
        }
        result.best_descriptor_matches = std::max(
            result.best_descriptor_matches,
            static_cast<int>(matches.size())
        );
        ranked.push_back({candidate, std::move(matches)});
    }

    std::sort(
        ranked.begin(), ranked.end(),
        [](const Ranked& a, const Ranked& b)
        {
            return a.matches.size() > b.matches.size();
        }
    );

    const std::size_t limit = std::min(shortlist_size_, ranked.size());
    for (std::size_t rank = 0; rank < limit; ++rank)
    {
        const auto& item = ranked[rank];
        LoopCandidate candidate;
        candidate.current_keyframe_id = current->id;
        candidate.candidate_keyframe_id = item.keyframe->id;
        candidate.descriptor_matches =
            static_cast<int>(item.matches.size());

        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;
        std::vector<std::size_t> match_indices;
        for (std::size_t i = 0; i < item.matches.size(); ++i)
        {
            const auto& match = item.matches[i];
            const auto association = item.keyframe
                ->feature_to_mappoint.find(
                    static_cast<std::size_t>(match.trainIdx)
                );
            if (association == item.keyframe->feature_to_mappoint.end())
                continue;
            const auto point = local_map.getMapPoint(association->second);
            if (!point || !point->active || !point->position_w.allFinite())
                continue;
            const Eigen::Vector4d point_candidate =
                item.keyframe->T_W_C.inverse() *
                Eigen::Vector4d(
                    point->position_w.x(), point->position_w.y(),
                    point->position_w.z(), 1.0
                );
            if (!point_candidate.allFinite() || point_candidate.z() <= 1e-8)
                continue;
            if (match.queryIdx < 0 ||
                static_cast<std::size_t>(match.queryIdx) >=
                    current->features.keypoints.size())
                continue;
            object_points.emplace_back(
                static_cast<float>(point->position_w.x()),
                static_cast<float>(point->position_w.y()),
                static_cast<float>(point->position_w.z())
            );
            image_points.push_back(
                current->features.keypoints[match.queryIdx].pt
            );
            match_indices.push_back(i);
        }
        candidate.usable_3d2d = static_cast<int>(object_points.size());

        if (candidate.descriptor_matches >= 30 &&
            candidate.usable_3d2d >= 15)
        {
            PnPSolver solver;
            const auto pose = solver.estimate(
                object_points,
                image_points,
                camera
            );
            candidate.geometric_inliers = pose.pnp_inlier_count;
            candidate.inlier_ratio = candidate.usable_3d2d > 0
                ? static_cast<double>(candidate.geometric_inliers) /
                  candidate.usable_3d2d
                : 0.0;

            if (pose.success && candidate.geometric_inliers >= 20 &&
                candidate.inlier_ratio >= 0.70 &&
                pose.rotation_matrix.rows == 3 &&
                pose.rotation_matrix.cols == 3)
            {
                Eigen::Matrix4d T_C_W = Eigen::Matrix4d::Identity();
                for (int r = 0; r < 3; ++r)
                {
                    for (int c = 0; c < 3; ++c)
                        T_C_W(r, c) = pose.rotation_matrix.at<double>(r, c);
                    T_C_W(r, 3) = pose.tvec.at<double>(r, 0);
                }
                const Eigen::Matrix4d T_W_C_current = T_C_W.inverse();
                candidate.T_candidate_current =
                    item.keyframe->T_W_C.inverse() * T_W_C_current;
                candidate.verified =
                    candidate.T_candidate_current.allFinite() &&
                    candidate.T_candidate_current.block<3, 3>(0, 0)
                        .determinant() > 0.99 &&
                    relativeRotationDegrees(
                        candidate.T_candidate_current
                    ) < 120.0 &&
                    candidate.T_candidate_current
                        .block<3, 1>(0, 3).norm() < 20.0;
            }
        }

        result.best_geometric_inliers = std::max(
            result.best_geometric_inliers,
            candidate.geometric_inliers
        );
        ++result.candidates_tested;
        if (candidate.verified)
            result.verified_loops.push_back(candidate);
        else
            ++result.rejected;
        result.candidates.push_back(candidate);
    }

    return result;
}

} // namespace my_slam
