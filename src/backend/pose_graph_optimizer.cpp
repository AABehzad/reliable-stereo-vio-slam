#include "my_slam/backend/pose_graph_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace my_slam
{
namespace
{

struct PoseBlock { double data[6] = {}; };

Eigen::Vector3d rotationVector(const Eigen::Matrix3d& rotation)
{
    const Eigen::AngleAxisd angle_axis(rotation);
    if (angle_axis.angle() < 1e-12)
        return Eigen::Vector3d::Zero();
    return angle_axis.axis() * angle_axis.angle();
}

PoseBlock poseToBlock(const Eigen::Matrix4d& pose)
{
    PoseBlock block;
    const Eigen::Vector3d rotation =
        rotationVector(pose.block<3, 3>(0, 0));
    for (int i = 0; i < 3; ++i)
    {
        block.data[i] = rotation(i);
        block.data[i + 3] = pose(i, 3);
    }
    return block;
}

Eigen::Matrix4d blockToPose(const PoseBlock& block)
{
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    const Eigen::Vector3d rotation(
        block.data[0], block.data[1], block.data[2]
    );
    const double angle = rotation.norm();
    if (angle > 1e-12)
        pose.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(angle, rotation / angle).toRotationMatrix();
    for (int i = 0; i < 3; ++i) pose(i, 3) = block.data[i + 3];
    return pose;
}

struct EdgeCost
{
    EdgeCost(const Eigen::Matrix4d& measurement, double weight)
        : weight_(weight),
          translation_(measurement.block<3, 1>(0, 3)),
          rotation_(rotationVector(measurement.block<3, 3>(0, 0)))
    {
        ceres::AngleAxisToQuaternion(
            rotation_.data(), measurement_quaternion_
        );
    }

    template <typename T>
    bool operator()(const T* const from, const T* const to,
                    T* residuals) const
    {
        T from_q[4], to_q[4];
        ceres::AngleAxisToQuaternion(from, from_q);
        ceres::AngleAxisToQuaternion(to, to_q);
        T inverse_from_q[4] = {
            from_q[0], -from_q[1], -from_q[2], -from_q[3]
        };
        T relative_q[4];
        ceres::QuaternionProduct(inverse_from_q, to_q, relative_q);
        T measurement_q[4] = {
            T(measurement_quaternion_[0]), T(measurement_quaternion_[1]),
            T(measurement_quaternion_[2]), T(measurement_quaternion_[3])
        };
        T inverse_measurement_q[4] = {
            measurement_q[0], -measurement_q[1],
            -measurement_q[2], -measurement_q[3]
        };
        T rotation_error[4];
        ceres::QuaternionProduct(
            inverse_measurement_q, relative_q, rotation_error
        );

        const T inverse_from_rotation[3] = {
            -from[0], -from[1], -from[2]
        };
        const T translation_delta[3] = {
            to[3] - from[3], to[4] - from[4], to[5] - from[5]
        };
        T relative_translation[3];
        ceres::AngleAxisRotatePoint(
            inverse_from_rotation, translation_delta, relative_translation
        );
        for (int i = 0; i < 3; ++i)
        {
            residuals[i] = T(weight_) *
                (relative_translation[i] - T(translation_(i)));
            residuals[i + 3] = T(2.0 * weight_) * rotation_error[i + 1];
        }
        return true;
    }

    double weight_;
    Eigen::Vector3d translation_;
    Eigen::Vector3d rotation_;
    double measurement_quaternion_[4] = {};
};

double trajectoryLength(
    const std::vector<KeyFrameId>& ids,
    const std::unordered_map<KeyFrameId, Eigen::Matrix4d>& poses)
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
    const std::vector<LoopConstraint>& constraints) const
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
        blocks.emplace(id, poseToBlock(keyframe->T_W_C));
    }
    if (ids.size() < 2 || constraints.empty()) return result;
    result.nodes = ids.size();
    result.trajectory_length_before = trajectoryLength(ids, old_poses);

    ceres::Problem problem;
    for (auto& entry : blocks)
        problem.AddParameterBlock(entry.second.data, 6);

    for (std::size_t i = 1; i < ids.size(); ++i)
    {
        const Eigen::Matrix4d measurement =
            old_poses[ids[i - 1]].inverse() * old_poses[ids[i]];
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<EdgeCost, 6, 6, 6>(
                new EdgeCost(measurement, 1.0)),
            new ceres::HuberLoss(1.0),
            blocks[ids[i - 1]].data, blocks[ids[i]].data);
        ++result.odometry_edges;
    }

    for (const auto& constraint : constraints)
    {
        if (blocks.find(constraint.candidate_keyframe_id) == blocks.end() ||
            blocks.find(constraint.current_keyframe_id) == blocks.end() ||
            !constraint.T_candidate_current.allFinite())
            continue;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<EdgeCost, 6, 6, 6>(
                new EdgeCost(constraint.T_candidate_current, 0.5)),
            new ceres::HuberLoss(1.0),
            blocks[constraint.candidate_keyframe_id].data,
            blocks[constraint.current_keyframe_id].data);
        ++result.loop_edges;
    }
    if (result.loop_edges == 0) return result;

    problem.SetParameterBlockConstant(blocks[ids.front()].data);
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.iterations = summary.iterations.size();
    result.success = summary.IsSolutionUsable();
    if (!result.success) return result;

    std::unordered_map<KeyFrameId, Eigen::Matrix4d> new_poses;
    double total_translation = 0.0;
    double total_rotation = 0.0;
    for (const auto id : ids)
    {
        new_poses[id] = blockToPose(blocks[id]);
        const Eigen::Matrix4d correction =
            old_poses[id].inverse() * new_poses[id];
        const double translation = correction.block<3, 1>(0, 3).norm();
        const double rotation = rotationVector(
            correction.block<3, 3>(0, 0)
        ).norm() * 180.0 / 3.14159265358979323846;
        total_translation += translation;
        total_rotation += rotation;
        result.max_translation_correction = std::max(
            result.max_translation_correction, translation);
        result.max_rotation_correction_deg = std::max(
            result.max_rotation_correction_deg, rotation);
        const auto keyframe = local_map.getKeyFrame(id);
        if (keyframe) keyframe->T_W_C = new_poses[id];
    }
    result.mean_translation_correction =
        total_translation / static_cast<double>(ids.size());
    result.mean_rotation_correction_deg =
        total_rotation / static_cast<double>(ids.size());

    for (const auto& map_point : local_map.activeMapPoints())
    {
        if (!map_point) continue;
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        std::size_t count = 0;
        for (const auto& observation : map_point->observations)
        {
            const auto old_pose = old_poses.find(observation.first);
            const auto new_pose = new_poses.find(observation.first);
            if (old_pose == old_poses.end() || new_pose == new_poses.end())
                continue;
            const Eigen::Vector4d point(
                map_point->position_w.x(), map_point->position_w.y(),
                map_point->position_w.z(), 1.0);
            sum += (new_pose->second * old_pose->second.inverse() * point)
                .head<3>();
            ++count;
        }
        if (count > 0)
            map_point->position_w = sum / static_cast<double>(count);
    }
    result.trajectory_length_after = trajectoryLength(ids, new_poses);
    return result;
}

} // namespace my_slam
