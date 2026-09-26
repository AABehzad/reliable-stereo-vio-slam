cmake_minimum_required(VERSION 3.16)

project(
    my_slam
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)

# =========================================
# SLAM Core Library
# =========================================

add_library(
    my_slam_core

    src/io/euroc_reader.cpp

    src/camera/camera_model.cpp

    src/frontend/orb_extractor.cpp
    src/frontend/feature_matcher.cpp
    src/frontend/local_map_tracker.cpp

    src/geometry/two_view_geometry.cpp
    src/geometry/stereo_geometry.cpp
    src/geometry/pnp_solver.cpp

    src/map/local_map.cpp
    src/backend/local_bundle_adjuster.cpp

)

target_include_directories(
    my_slam_core

    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    my_slam_core

    PUBLIC
        ${OpenCV_LIBS}
        Eigen3::Eigen
        yaml-cpp
        Ceres::ceres
)

# =========================================
# Smoke Test
# =========================================

add_executable(
    smoke_test
    apps/smoke_test.cpp
)

target_link_libraries(
    smoke_test
    PRIVATE
        my_slam_core
)

# =========================================
# EuRoC Application
# =========================================

add_executable(
    run_keyframe_vo
    apps/run_keyframe_vo.cpp
)

target_link_libraries(
    run_keyframe_vo
    PRIVATE
        my_slam_core
)