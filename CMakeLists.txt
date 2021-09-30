cmake_minimum_required(VERSION 3.10)
project(GVIO)

SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_FLAGS_Release "${CMAKE_C_FLAGS_Release} -O3")

set(CMAKE_CXX_STANDARD 14)

find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
MESSAGE("DEBUG DEBUG DEBUG DEBUG")
MESSAGE(${CMAKE_CURRENT_SOURCE_DIR})
LINK_DIRECTORIES(/usr/local/lib /usr/local)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty
        ${CMAKE_CURRENT_SOURCE_DIR}/includes)

add_executable(GVIO IMUKittiGPS.cpp srcs/feature_tracker.cpp srcs/parameters.cpp srcs/load_kitti.cpp includes/project_factor.h includes/SlideWindow.h includes/ZUPT.h includes/gps_factor_with_lever_arm_calibration.h)
target_link_libraries(GVIO gtsam gtsam_unstable ${OpenCV_LIBS})
