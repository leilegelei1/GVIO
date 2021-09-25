//
// Created by jerry on 2021/9/19.
//

#ifndef GVIO_BASESTRUCT_H
#define GVIO_BASESTRUCT_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <gtsam/navigation/CombinedImuFactor.h>
using namespace std;
using namespace gtsam;

struct KittiCalibration {
    double body_ptx;
    double body_pty;
    double body_ptz;
    double body_prx;
    double body_pry;
    double body_prz;
    double accelerometer_sigma;
    double gyroscope_sigma;
    double integration_sigma;
    double accelerometer_bias_sigma;
    double gyroscope_bias_sigma;
    double average_delta_t;
};

struct ImuMeasurement {
    double time;
    double dt;
    Vector3 accelerometer;
    Vector3 gyroscope;  // omega
};

struct GpsMeasurement {
    double time;
    Vector3 position;  // x,y,z
};

struct ImageMeasurement{
    double time;
    string path;
};

struct Track{
    int id;
    bool active; //判断点是否被激活
    vector<pair<int,int>> imgIdAndPtID;
};

struct Frame{
    int frame_id;
    cv::Mat img;

    vector<cv::Point2f> undistort_pts; //记录2D点位置
    vector<cv::Point2f> distort_pts;

    vector<int> host_track; //记录本帧为主导的track id
    unordered_map<int,int> pt2track; //记录2D点id到trackid
};

void loadKittiData(KittiCalibration& kitti_calibration,
                   vector<ImuMeasurement>& imu_measurements,
                   vector<GpsMeasurement>& gps_measurements,
                   vector<ImageMeasurement>& img_measurements,
                   string imu_metadata_file,
                   string imu_data_file,
                   string gps_data_file,
                   string img_data_file);

#endif //GVIO_BASESTRUCT_H
