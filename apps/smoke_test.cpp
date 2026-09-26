#include <iostream>

#include <opencv2/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/version.h>

#include <yaml-cpp/yaml.h>

int main()
{
    std::cout << "=================================\n";
    std::cout << "        My SLAM Environment\n";
    std::cout << "=================================\n\n";

    // --------------------------------------------------
    // OpenCV
    // --------------------------------------------------

    std::cout
        << "OpenCV version : "
        << CV_VERSION
        << '\n';

    // --------------------------------------------------
    // Eigen
    // --------------------------------------------------

    Eigen::Matrix3d rotation =
        Eigen::Matrix3d::Identity();

    Eigen::Vector3d position(
        1.0,
        2.0,
        3.0
    );

    std::cout
        << "\nEigen rotation matrix:\n"
        << rotation
        << '\n';

    std::cout
        << "\nEigen position:\n"
        << position.transpose()
        << '\n';

    // --------------------------------------------------
    // Ceres
    // --------------------------------------------------

    std::cout
        << "\nCeres version  : "
        << CERES_VERSION_STRING
        << '\n';

    // --------------------------------------------------
    // YAML
    // --------------------------------------------------

    YAML::Node config =
        YAML::Load(
            "camera: cam0\n"
            "fps: 20\n"
        );

    std::cout
        << "YAML camera    : "
        << config["camera"].as<std::string>()
        << '\n';

    std::cout
        << "YAML FPS       : "
        << config["fps"].as<int>()
        << '\n';

    std::cout << "\n=================================\n";
    std::cout << "Environment is ready for SLAM.\n";
    std::cout << "=================================\n";

    return 0;
}