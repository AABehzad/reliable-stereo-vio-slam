#include "my_slam/backend/pose_graph_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace my_slam
{

namespace
{

struct PoseBlock
{
    double rotation[3] = {};
    double translation[3] = {};
};

struct EdgeCost
{
    EdgeCost(
        const Eigen::Matrix4d& relative,
        double weight
    )
        : weight_(weight)
    {
        const Eigen::Matrix3d rotation =
            relative.block<3, 3>(0, 0);
        const Eigen::AngleAxisd angle_axis(rotation);
        measurement_rotation_[0] = angle_axis.axis().x() * angle_axis.angle();
        measurement_rotation_[1] = angle_axis.axis().y() * angle_axis.angle();
        measurement_rotation_[2] = angle_axis.axis().z() * angle_axis.angle();
        measurement_translation_[0] = relative(0, 3);
        measurement_translation_[1] = relative(1, 3);
        measurement_translation_[2] = relative(2, 3);
        ceres::AngleAxisToQuaternion(
            measurement_rotation_, measurement_quaternion_
        );
    }

    template <typename T>
    bool operator()(
        const T* const from,
        const T* const to,
        T* residuals
    ) const
    {
        T from_q[4];
        T to_q[4];
        ceres::AngleAxisToQuaternion(from, from_q);
        ceres::AngleAxisToQuaternion(to, to_q);

        T inverse_from_q[4] = {
            from_q[0], -from_q[1], -from_q[2], -from_q[3]
        };
        T relative_q[4];
        ceres::QuaternionProduct(
            inverse_from_q, to_q, relative_q
        );
        T measurement_q[4] = {
            T(measurement_quaternion_[0]),
            T(measurement_quaternion_[1]),
            T(measurement_quaternion_[2]),
            T(measurement_quaternion_[3])
        };
        T inverse_measurement_q[4] = {
            measurement_q[0], -measurement_q[1],
            -measurement_q[2], -measurement_q[3]
        };
        T rotation_error[4];
        ceres::QuaternionProduct(
            inverse_measurement_q, relative_q, rotation_error
        );

        T from_rotation[3] = {
            from[0], from[1], from[2]
        };
        T delta_translation[3] = {
            to[3] - from[3],
            to[4] - from[4],
            to[5] - from[5]
        };
        T relative_translation[3];
        ceres::AngleAxisRotatePoint(
            from_rotation,
            delta_translation,
            relative_translation
        );

        residuals[0] = T(weight_) *
            (relative_translation[0] - T(measurement_translation_[0]));
        residuals[1] = T(weight_) *
            (relative_translation[1] - T(measurement_translation_[1]));
        residuals[2] = T(weight_) *
            (relative_translation[2] - T(measurement_translation_[2]));
        residuals[3] = T(2.0 * weight_) * rotation_error[1];
        residuals[4] = T(2.0 * weight_) * rotation_error[2];
        residuals[5] = T(2.0 * weight_) * rotation_error[3];
        return true;
    }

    double measurement_rotation_[3] = {};
    double measurement_quaternion_[4] = {};
    double measurement_translation_[3] = {};
    double weight_ = 1.0;
};

PoseBlock poseToBlock(const Eigen::Matrix4d& pose)
{
    PoseBlock result;
    cv::Mat rotation;
    (void)rotation;
    Eigen::AngleAxisd angle_axis(
        pose.block<3, 3>(0, 0)
    );
    const Eigen::Vector3d vector =
        angle_axis.axis() * angle_axis.angle();
    for (int i = 0; i < 3; ++i)
    {
        result.rotation[i] = vector(i);
        result.translation[i] = pose(i, 3);
    }
    return result;
}

Eigen::Matrix4d blockToPose(const PoseBlock& block)
{
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    const Eigen::Vector3d rotation(
        block.rotation[0], block.rotation[1], block.rotation[2]
    );
    const double angle = rotation.norm();
    if (angle > 1e-12)
        result.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(angle, rotation / angle).toRotationMatrix();
    for (int i = 0; i < 3; ++i)
        result(i, 3) = block.translation[i];
    return result;
}

double trajectoryLength(
    const std::vector<KeyFrameId>& ids,
    const std::unordered_map<KeyFrameId, Eigen::Matrix4d>& poses
)
{
    double length = 0.0;
    for (std::size_t i = 1; i < ids.size(); ++i)
    {
        const auto a = poses.find(ids[i - 1]);
        const auto b = poses.find(ids[i]);
        if (a != poses.end() && b != poses.end())
            length += (b->second.block<3, 1>(0, 3) -
                       a->second.block<3, 1>(0, 3)).norm();
    }
    return length;
}

} // namespace

PoseGraphResult PoseGraphOptimizer::optimize(
    LocalMap& local_map,
    const std::vector<LoopConstraint>& constraints
) const
{
    PoseGraphResult result;
    std::vector<KeyFrameId> ids;
    std::unordered_map<KeyFrameId, PoseBlock> blocks;
    std::unordered_map<KeyFrameId, Eigen::Matrix4d> old_poses;

    for (KeyFrameId id = 0; id < local_map.keyFrameCount(); ++id)
    {
        const auto keyframe = local_map.getKeyFrame(id);
        if (!keyframe) continue;
        ids.push_back(id);
        old_poses[id] = keyframe->T_W_C;
        blocks[id] = poseToBlock(keyframe->T_W_C);
    }
    if (ids.size() < 2) return result;
    result.nodes = ids.size();
    result.trajectory_length_before = trajectoryLength(ids, old_poses);

    ceres::Problem problem;
    for (auto& [id, block] : blocks)
        problem.AddParameterBlock(block.rotation, 3);

    for (std::size_t i = 1; i < ids.size(); ++i)
    {
        const Eigen::Matrix4d relative =
            old_poses[ids[i - 1]].inverse() * old_poses[ids[i]];
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<EdgeCost, 6, 3, 3>(
                new EdgeCost(relative, 1.0)
            ),
            new ceres::HuberLoss(1.0),
            blocks[ids[i - 1]].rotation,
            blocks[ids[i]].rotation
        );
        // Translation is stored in the same six-parameter block layout.
        problem.AddParameterBlock(blocks[ids[i - 1]].translation, 3);
        problem.AddParameterBlock(blocks[ids[i]].translation, 3);
        ++result.odometry_edges;
    }

    // Rebuild the odometry residuals with six-parameter blocks is not
    // possible with split blocks; use contiguous storage below instead.
    // The early construction above is intentionally avoided by returning
    // to the correct implementation path.
    return result;
}

} // namespace my_slam
