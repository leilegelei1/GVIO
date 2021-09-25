/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file IMUKittiExampleGPS
 * @brief Example of application of ISAM2 for GPS-aided navigation on the KITTI VISION BENCHMARK SUITE
 * @author Ported by Thomas Jespersen (thomasj@tkjelectronics.dk), TKJ Electronics
 */


// GTSAM related includes.
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <baseStruct.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <gps_facor.h>
#include <parameters.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/geometry/Point2.h>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <feature_tracker.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <iomanip>
#include <algorithm>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Marginals.h>

using namespace std;
using namespace gtsam;

//应用给GPS的factor
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::L;  // landmark (x,y,z)

const string pose_output = "pose.csv";
const string pts_output = "pts.csv";
const string infer_output = "infer.csv";

Point3 myTriangulatePoint3(const std::vector<Pose3> &poses,
                           boost::shared_ptr<Cal3_S2> sharedCal,
                           const Point2Vector &measurements, double rank_tol = 1e-9) {
    std::vector<Matrix34, Eigen::aligned_allocator<Matrix34>> projection_matrices;
    CameraProjectionMatrix<Cal3_S2> createP(*sharedCal); // partially apply
    for (const Pose3 &pose: poses)
        projection_matrices.push_back(createP(pose));

    // Triangulate linearly
    Point3 point = triangulateDLT(projection_matrices, measurements, rank_tol);
    return point;
}

void showTracking(cv::Mat& img,vector<cv::Point2f>& initialized,vector<cv::Point2f>& uninitialized,vector<string>& track_len,vector<cv::Point2f>& marged_pts)
{
    cv::Mat gray_color;
    cv::cvtColor(img,gray_color,cv::COLOR_GRAY2BGR);
    for(auto pt : initialized)
        cv::circle(gray_color,pt,5,cv::Scalar(0,0,255),-1);

    for(int ix=0;ix < uninitialized.size();ix++)
    {
        auto pt = uninitialized[ix];
        cv::circle(gray_color,pt,5,cv::Scalar(255,0,0),-1);
        stringstream ss;
        ss << track_len[ix];
//        /cout << track_len[ix] << endl;
        cv::putText(gray_color,ss.str(),cv::Point(pt.x,pt.y),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(0,255,0));
    }

//    for(auto pt : uninitialized)
//        cv::circle(gray_color,pt,5,cv::Scalar(255,0,0),-1);
    for(auto pt : marged_pts)
        cv::circle(gray_color,pt,5,cv::Scalar(0,255,0),-1);
    cv::namedWindow("track");
    cv::imshow("track",gray_color);
    cv::waitKey(2);
}

void showTwoPts(cv::Mat& cur_img,cv::Mat& forw_img,std::vector<cv::Point2f>& cur_pts,std::vector<cv::Point2f>& forw_pts)
{
    vector<cv::KeyPoint> keyPoint1, keyPoint2;
    vector<cv::DMatch> matchePoints;
    for(int i=0;i<cur_pts.size();i++)
    {
        cv::KeyPoint kpt1,kpt2;
        kpt1.pt = cur_pts[i];
        kpt2.pt = forw_pts[i];
        keyPoint1.emplace_back(kpt1);
        keyPoint2.emplace_back(kpt2);
        cv::DMatch match(i,i,0);
        matchePoints.emplace_back(match);
    }

    cv::Mat img_match;
    cv::drawMatches(cur_img, keyPoint1, forw_img, keyPoint2, matchePoints, img_match);
    cv::namedWindow("match", cv::WINDOW_FULLSCREEN);
    imshow("match",img_match);
    cv::waitKey(0);
}

int main() {

    KittiCalibration kitti_calibration;
    vector<ImuMeasurement> imu_measurements;
    vector<GpsMeasurement> gps_measurements;
    vector<ImageMeasurement> img_measurements;
    loadKittiData(kitti_calibration, imu_measurements, gps_measurements, img_measurements,
                  "/home/jerry/Desktop/GVIO/kittiData/KittiEquivBiasedImu_metadata.txt",
                  "/home/jerry/Downloads/2011_10_03_drive_0027_extract/2011_10_03/2011_10_03_drive_0027_extract/converted/kitti_imus.txt",
                  "/home/jerry/Downloads/2011_10_03_drive_0027_extract/2011_10_03/2011_10_03_drive_0027_extract/converted/kitti_gps.txt",
                  "/home/jerry/Downloads/2011_10_03_drive_0027_extract/2011_10_03/2011_10_03_drive_0027_extract/converted/kitti_imgs.txt");

    readParameters("/home/jerry/Desktop/GVIO/configs/kitti.yaml");

    FILE *fp_pose = fopen(pose_output.c_str(), "w+");
    setbuf(fp_pose,NULL);
    FILE *fp_pts = fopen(pts_output.c_str(), "w+");
    setbuf(fp_pts,NULL);

    FILE *infer_pts = fopen(infer_output.c_str(), "w+");
    setbuf(infer_pts,NULL);

    unordered_map<int, int> valid_l_idx, valid_x_idx; //记录仍然存在的landmark和位姿状态

    //
    Vector6 BodyP = (Vector6() << kitti_calibration.body_ptx, kitti_calibration.body_pty, kitti_calibration.body_ptz,
            kitti_calibration.body_prx, kitti_calibration.body_pry, kitti_calibration.body_prz)
            .finished();
    auto body_T_imu = Pose3::Expmap(BodyP);
    if (!body_T_imu.equals(Pose3(), 1e-5)) {
        printf("Currently only support IMUinBody is identity, i.e. IMU and body frame are the same");
        exit(-1);
    }

    Pose3 C2I(Rot3(RIC[0]), TIC[0]);

    Eigen::Matrix3d cam_k_eigen;
    cv::cv2eigen(cam_K, cam_k_eigen);
    gtsam::Cal3_S2::shared_ptr cam_K_gtsam(new Cal3_S2(FX, FY, 0.0, CX, CY));

    auto noise_model_gps_new = noiseModel::Isotropic::Sigma(3, 0.001);

    auto noise_model_pt2d = noiseModel::Isotropic::Sigma(2, 15.0);  // 2D 图像观测误差模型
    auto noise_model_pt2d_robust = noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(4),
                                                              noise_model_pt2d);

//    auto noise_model_pt2d_robust = noiseModel::Isotropic::Sigma(2, 4.0);

    size_t gps_counter = 0, img_counter = 0, imu_counter = 0;
    size_t frame_counter = 0;
    bool initialized = false;

    //GPS 和 图像分成两个窗口存储
    vector<int> gps_sw;
    vector<int> img_sw;

    double g = 9.8;

    auto w_coriolis = Vector3::Zero();  // zero vector
    auto current_bias = imuBias::ConstantBias();  // init with zero bias
    auto sigma_init_x = noiseModel::Diagonal::Precisions((Vector6() << Vector3::Constant(0),
            Vector3::Constant(1.0)).finished());
    auto sigma_init_v = noiseModel::Diagonal::Sigmas(Vector3::Constant(1000.0));
    auto sigma_init_b = noiseModel::Diagonal::Sigmas((Vector6() << Vector3::Constant(0.100),
            Vector3::Constant(5.00e-05))
                                                             .finished());
    // Set IMU preintegration parameters
    double scale = 1;
    Matrix33 measured_acc_cov = I_3x3 * pow(kitti_calibration.accelerometer_sigma * scale, 2);
    Matrix33 measured_omega_cov = I_3x3 * pow(kitti_calibration.gyroscope_sigma * scale, 2);
    // error committed in integrating position from velocities
    Matrix33 integration_error_cov = I_3x3 * pow(kitti_calibration.integration_sigma * scale, 2);

    auto imu_params = PreintegratedImuMeasurements::Params::MakeSharedU(g);
    imu_params->accelerometerCovariance = measured_acc_cov;     // acc white noise in continuous
    imu_params->integrationCovariance = integration_error_cov;  // integration uncertainty continuous
    imu_params->gyroscopeCovariance = measured_omega_cov;       // gyro white noise in continuous
    imu_params->omegaCoriolis = w_coriolis;

    FeatureTracker* tracker = new FeatureTracker();
    std::shared_ptr<PreintegratedImuMeasurements> current_summarized_measurement = nullptr;
    //程序的主循环
    NonlinearFactorGraph graph;
    Values values;
    bool first_frame = true;
    double t_previous = 0;

    vector<pair<int, int>> gps2imu_recorder;
    unordered_map<int, int> img_frame_id2global_frame_id;
    bool processImg = true;

    while (gps_counter < gps_measurements.size() && img_counter < img_measurements.size() &&
           imu_counter < imu_measurements.size()) {
        if (!initialized)//尚未初始化
        {
//            cout << "final error = " << graph.error(values) << endl;
            if (first_frame) {
                //清空所有
                gps_sw.clear();
                img_sw.clear();
                values.clear();
                gps2imu_recorder.clear();
                graph.resize(0);

                gps2imu_recorder.clear();
                img_frame_id2global_frame_id.clear();
                valid_l_idx.clear();
                valid_x_idx.clear();
                cout << "tracker!";
                delete tracker;
                tracker = new FeatureTracker();
                cout << "tracker" << endl;
                auto gps_time = gps_measurements[gps_counter].time; //第一帧从GPS帧开始
                while (img_measurements[img_counter].time < gps_time)
                    img_counter++;
                while (imu_measurements[imu_counter].time <= gps_time)
                    imu_counter++;
                imu_counter--;
                first_frame = false;

                auto current_pose_global = Pose3(Rot3(), gps_measurements[gps_counter].position);
                Vector3 current_velocity_global = Vector3::Zero();

                values.insert(X(frame_counter), current_pose_global);
                values.insert(V(frame_counter), current_velocity_global);
                values.insert(B(frame_counter), current_bias);
                graph.emplace_shared<PriorFactor<Pose3>>(X(frame_counter), current_pose_global, sigma_init_x);
                graph.emplace_shared<PriorFactor<Vector3>>(V(frame_counter), current_velocity_global, sigma_init_v);
                graph.emplace_shared<PriorFactor<imuBias::ConstantBias>>(B(frame_counter), current_bias, sigma_init_b);

                valid_x_idx[frame_counter] = 0;

                gps2imu_recorder.push_back(make_pair(gps_counter, imu_counter));

                gps_sw.emplace_back(frame_counter);
                gps_counter++;
                frame_counter++;
                t_previous = gps_time;
                continue;
            } else {
                double gps_time = gps_measurements[gps_counter].time;
                double img_time = img_measurements[img_counter].time;

                cout << fixed << setprecision(9) << gps_time << " " << img_time << " " << t_previous << endl;
                cout << "after error = " << graph.error(values) << endl;


                current_summarized_measurement = std::make_shared<PreintegratedImuMeasurements>(imu_params,
                                                                                                current_bias);
                static size_t included_imu_measurement_count = 0;
                while (imu_counter < imu_measurements.size() && imu_measurements[imu_counter].time <= gps_time) {
                    if (imu_measurements[imu_counter].time >= t_previous) {

                        current_summarized_measurement->integrateMeasurement(
                                imu_measurements[imu_counter].accelerometer,
                                imu_measurements[imu_counter].gyroscope,
                                imu_measurements[imu_counter].dt);
                        included_imu_measurement_count++;
                    }
                    imu_counter++;
                }
                gps2imu_recorder.push_back(make_pair(gps_counter, imu_counter));

                graph.emplace_shared<ImuFactor>(X(frame_counter - 1), V(frame_counter - 1),
                                                X(frame_counter), V(frame_counter),
                                                B(frame_counter - 1), *current_summarized_measurement);
                auto sigma_between_b = noiseModel::Diagonal::Sigmas((Vector6() <<
                                                                               Vector3::Constant(
                                                                                       sqrt(included_imu_measurement_count) *
                                                                                       kitti_calibration.accelerometer_bias_sigma),
                        Vector3::Constant(
                                sqrt(included_imu_measurement_count) * kitti_calibration.gyroscope_bias_sigma))
                                                                            .finished());
                graph.emplace_shared<BetweenFactor<imuBias::ConstantBias>>(B(frame_counter - 1),
                                                                           B(frame_counter),
                                                                           imuBias::ConstantBias(),
                                                                           sigma_between_b);
                auto gps_pose_new = Point3(gps_measurements[gps_counter].position);
                graph.emplace_shared<GpsFactor<Pose3>>(X(frame_counter), gps_pose_new, noise_model_gps_new);

                NavState predict;
                predict = current_summarized_measurement->predict(
                        NavState(values.at(X(frame_counter - 1)).cast<Pose3>(),
                                 values.at(V(frame_counter - 1)).cast<Velocity3>()), current_bias);

                auto gps_pose = Pose3(Rot3(predict.R()), gps_measurements[gps_counter].position);
                values.insert(X(frame_counter), gps_pose);
                Vector3 current_velocity_global = Vector3::Zero();
                values.insert(V(frame_counter), predict.velocity());
                values.insert(B(frame_counter), current_bias);

                valid_x_idx[frame_counter] = 0;

                gps_sw.emplace_back(frame_counter);
                gps_counter++;
                frame_counter++;
                t_previous = gps_time;
            }
            if (gps_sw.size() >= 8) {
                //先求解初始化问题 然后再三角化构建landmark整体优化
                cout << "gps initial error = " << graph.error(values) << endl;
                values = LevenbergMarquardtOptimizer(graph, values).optimize();
                cout << "gps after error = " << graph.error(values) << endl;

                // TODO 之前那样处理太过于麻烦了还不如直接给每个GPS找一个最近的图像
                int counter = -1;
                for (auto i: gps2imu_recorder) {
                    ++counter;
                    int gps_counter = i.first, imu_counter = i.second;
                    double gps_time = gps_measurements[gps_counter + 1].time;
                    t_previous = gps_measurements[gps_counter].time;
                    while (img_measurements[img_counter].time < gps_time) //找到在GPS之后最近的一帧图像
                        img_counter++;
                    auto img_time = img_measurements[img_counter].time;

                    int gps_frame_id = gps_sw[counter];
                    current_bias = values.at<imuBias::ConstantBias>(B(gps_frame_id));

                    current_summarized_measurement = std::make_shared<PreintegratedImuMeasurements>(imu_params,
                                                                                                    current_bias);
                    static size_t included_imu_measurement_count = 0;
                    double total_Dt = 0;
                    while (imu_counter < imu_measurements.size() && imu_measurements[imu_counter].time <= img_time) {
                        if (imu_measurements[imu_counter].time >= t_previous) {
                            current_summarized_measurement->integrateMeasurement(
                                    imu_measurements[imu_counter].accelerometer,
                                    imu_measurements[imu_counter].gyroscope,
                                    imu_measurements[imu_counter].dt);
                            total_Dt += imu_measurements[imu_counter].dt;
                            included_imu_measurement_count++;
                        }
                        imu_counter++;
                    }


                    graph.emplace_shared<ImuFactor>(X(gps_frame_id), V(gps_frame_id),
                                                    X(frame_counter), V(frame_counter),
                                                    B(gps_frame_id), *current_summarized_measurement);
                    auto sigma_between_b = noiseModel::Diagonal::Sigmas((Vector6() <<
                                                                                   Vector3::Constant(
                                                                                           sqrt(included_imu_measurement_count) *
                                                                                           kitti_calibration.accelerometer_bias_sigma),
                            Vector3::Constant(
                                    sqrt(included_imu_measurement_count) * kitti_calibration.gyroscope_bias_sigma))
                                                                                .finished());
                    graph.emplace_shared<BetweenFactor<imuBias::ConstantBias>>(B(gps_frame_id),
                                                                               B(frame_counter),
                                                                               imuBias::ConstantBias(),
                                                                               sigma_between_b);
                    NavState predict;
                    predict = current_summarized_measurement->predict(NavState(values.at(X(gps_frame_id)).cast<Pose3>(),
                                                                               values.at(
                                                                                       V(gps_frame_id)).cast<Velocity3>()),
                                                                      values.at(
                                                                              B(gps_frame_id)).cast<imuBias::ConstantBias>());
                    auto img_pose = Pose3(Rot3(predict.R()), predict.t());

                    cout << gps_time << " " << total_Dt << endl;
                    cout << img_pose << endl;
                    cout << predict.velocity() << endl;

                    values.insert(X(frame_counter), img_pose);
                    values.insert(V(frame_counter), predict.velocity());
                    values.insert(B(frame_counter), current_bias);

                    valid_x_idx[frame_counter] = 0;

                    auto img_p = img_measurements[img_counter].path;
                    cv::Mat img = cv::imread(img_p, cv::IMREAD_GRAYSCALE);
                    tracker->readImage(img);
                    img_frame_id2global_frame_id[tracker->frame_id - 1] = frame_counter;
                    img_sw.emplace_back(frame_counter);
                    frame_counter++;
                    img_counter++;
                    t_previous = img_time;
                }

                //三角化所有的track TODO 检查视差的三角化
                int cnt = 0;
                int cnt_tri = 0;
                cout << "track num: " << tracker->all_tracks.size() << endl;
                for (auto track: tracker->all_tracks) {
                    if (track.imgIdAndPtID.size() < 3)
                        continue;
                    cnt += 1;
                    std::vector<Pose3> poses;
                    Point2Vector measurements;
                    vector<int> poseIdx;
                    vector<int> frame_idx;
                    for (auto pair: track.imgIdAndPtID) {
                        int frame_id = pair.first, pt_id = pair.second;
                        for (auto j: tracker->all_frames) {
                            if (j.frame_id == frame_id) {
                                Pose3 imu_pose = values.at(X(img_frame_id2global_frame_id[frame_id])).cast<Pose3>();
                                Pose3 cam_pose = imu_pose.compose(C2I);
                                poses.push_back(cam_pose);
                                measurements.push_back(Point2(j.undistort_pts[pt_id].x, j.undistort_pts[pt_id].y));
//                                measurements.push_back(Point2(j.distort_pts[pt_id].x, j.distort_pts[pt_id].y));
                                poseIdx.push_back(img_frame_id2global_frame_id[frame_id]);
                                frame_idx.emplace_back(frame_id);
                            }
                        }
                    }



                    if (poses.size() > 3) {

//                        std::vector<Pose3> poses_new;
//                        Point2Vector measurements_new;
//                        poses_new.push_back(poses[0]);
//                        measurements_new.push_back(measurements[0]);
//                        poses_new.push_back(poses[poses.size() - 1]);
//                        measurements_new.push_back(measurements[measurements.size() - 1]);

//                        {
//                            //TODO 可视化观察重建效果
//                            cout << tracker->framdid2idx[frame_idx[0]] << " " << tracker->framdid2idx[frame_idx[frame_idx.size()-1]] << endl;
//                            auto cur_img = tracker->all_frames[tracker->framdid2idx[frame_idx[0]]].img;
//                            auto forw_img = tracker->all_frames[tracker->framdid2idx[frame_idx[frame_idx.size()-1]]].img;
//                            std::vector<cv::Point2f> cur_pts({cv::Point2f(measurements_new[0].x(),measurements_new[0].y())});
//                            std::vector<cv::Point2f> forw_pts({cv::Point2f(measurements_new[1].x(),measurements_new[1].y())});
//                            showTwoPts(cur_img,forw_img,cur_pts,forw_pts);
//                        }

//                        auto pt3d = myTriangulatePoint3(poses_new, cam_K_gtsam, measurements_new);

                        auto pt3d = myTriangulatePoint3(poses, cam_K_gtsam, measurements);
                        auto pt_local = poses[poses.size()-1].transformTo(pt3d);
                        if(pt_local.z()>100) //太远的点也先干掉 因为很有可能是动态物建出来的点 并不好
                            continue;

                        track.active = true;

                        auto pointNoise = noiseModel::Isotropic::Sigma(3, 10);
                        graph.addPrior(L(track.id), pt3d, pointNoise);  // add directly to graph

                        values.insert(L(track.id), pt3d);
                        valid_l_idx[track.id] = 0;
                        for (int i = 0; i < poseIdx.size(); i++) {
                            Pose3 imu_pose = values.at(X(poseIdx[i])).cast<Pose3>();
                            Pose3 cam_pose = imu_pose.compose(C2I);
                            Point3 pt_local = cam_pose.transformTo(pt3d);
                            if (pt_local.z() < 0)
                                continue;
                            graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
                                    measurements[i], noise_model_pt2d_robust, X(poseIdx[i]), L(track.id), cam_K_gtsam,
                                    C2I);
                        }

                        cnt_tri++;
                    }
                }

                cout << cnt << endl;
                cout << cnt_tri << endl;
                cout << "initial error = " << graph.error(values) << endl;
                LevenbergMarquardtParams params;
                LevenbergMarquardtParams::SetCeresDefaults(&params);
                values = LevenbergMarquardtOptimizer(graph, values, params).optimize();
                cout << "final error = " << graph.error(values) << endl;

                processImg = false;
                initialized = true;

//                for(auto i : valid_x_idx) {
//                    Pose3 pose = values.at<Pose3>(X(i.first));
//                    auto tvec = pose.translation();
//                    cout << tvec << endl;
//                    fprintf(fp_pose, "%f,%f,%f\n", tvec.x(), tvec.y(), tvec.z());
//                    cout << "----------" << frame_counter << endl;
//                }

                cout << "============================ init finished " <<endl;
//                return true;
            }

        } else//TODO 初始化完成
        {
//            if(true)
            try
            {
                float tag = 0.05;
                bool visual_only = false;
                bool incorrect_angle = false;
                if (!processImg) //处理GPS帧
                {
                    while (gps_measurements[gps_counter].time < t_previous + tag) //找到在GPS之后最近的一帧图像
                        gps_counter++;

                    double gps_time = gps_measurements[gps_counter].time;

                    if(gps_time - t_previous > 0.2) //有一些GPS数据有问题要退回到纯视觉处理
                    {
                        processImg = true;
                        continue;
                    }

                    current_summarized_measurement = std::make_shared<PreintegratedImuMeasurements>(imu_params,
                                                                                                    current_bias);
                    static size_t included_imu_measurement_count = 0;
                    while (imu_counter < imu_measurements.size() && imu_measurements[imu_counter].time <= gps_time) {
                        if (imu_measurements[imu_counter].time >= t_previous) {
                            if (imu_measurements[imu_counter].dt > 0) {
//                        cout << imu_measurements[imu_counter].dt << endl;
                                current_summarized_measurement->integrateMeasurement(
                                        imu_measurements[imu_counter].accelerometer,
                                        imu_measurements[imu_counter].gyroscope,
                                        imu_measurements[imu_counter].dt);
                                included_imu_measurement_count++;
                            } else
                                cout << "dt < 0 ----------- " << imu_measurements[imu_counter].dt << endl;
                        }
                        imu_counter++;
                    }
                    graph.emplace_shared<ImuFactor>(X(frame_counter - 1), V(frame_counter - 1),
                                                    X(frame_counter), V(frame_counter),
                                                    B(frame_counter - 1), *current_summarized_measurement);

                    included_imu_measurement_count = 1;

                    auto sigma_between_b = noiseModel::Diagonal::Sigmas((Vector6() <<
                                                                                   Vector3::Constant(
                                                                                           sqrt(included_imu_measurement_count) *
                                                                                           kitti_calibration.accelerometer_bias_sigma),
                            Vector3::Constant(
                                    sqrt(included_imu_measurement_count) * kitti_calibration.gyroscope_bias_sigma))
                                                                                .finished());

                    graph.emplace_shared<BetweenFactor<imuBias::ConstantBias>>(B(frame_counter - 1),
                                                                               B(frame_counter),
                                                                               imuBias::ConstantBias(),
                                                                               sigma_between_b);

                    auto gps_pose_new = Point3(gps_measurements[gps_counter].position);
                    auto gpsfactor = GpsFactor<Pose3>(X(frame_counter), gps_pose_new, noise_model_gps_new);
                    graph.emplace_shared<GpsFactor<Pose3>>(X(frame_counter), gps_pose_new, noise_model_gps_new);


                    NavState predict;
                    predict = current_summarized_measurement->predict(
                            NavState(values.at(X(frame_counter - 1)).cast<Pose3>(),
                                     values.at(V(frame_counter - 1)).cast<Velocity3>()), current_bias);

                    auto gps_pose = Pose3(Rot3(predict.R()), gps_measurements[gps_counter].position);
                    values.insert(X(frame_counter), gps_pose);
                    Vector3 current_velocity_global = Vector3::Zero();
                    values.insert(V(frame_counter), predict.velocity());
                    values.insert(B(frame_counter), current_bias);


//                    cout << "---" << endl;
//                    cout << gpsfactor.evaluateError(values.at<Pose3>(X(frame_counter))) << endl;
//                    cout << "---" << endl;

                    LevenbergMarquardtParams params;
                    LevenbergMarquardtParams::SetCeresDefaults(&params);
                    cout << "SW process gps" << endl;
//                    cout << "initial error = " << graph.error(values) << endl;
                    values = LevenbergMarquardtOptimizer(graph, values, params).optimize();
//                    cout << "final error = " << graph.error(values) << endl;

//                    cout << values.at<Pose3>(X(frame_counter)).translation() - gps_pose_new << endl;

                    gps_sw.emplace_back(frame_counter);
                    valid_x_idx[frame_counter] = 0;
                    frame_counter++;
                    gps_counter++;
                    t_previous = gps_time;

                    KeyVector marg_keys;
                    int front_idx = gps_sw[0];
                    marg_keys.push_back(X(front_idx));
                    marg_keys.push_back(V(front_idx));
                    marg_keys.push_back(B(front_idx));


                    graph = BatchFixedLagSmoother::CalculateMarginalFactors(graph, values, marg_keys);

                    Pose3 pose = values.at<Pose3>(X(frame_counter - 1));
                    auto tvec = pose.translation();
                    cout << tvec.transpose() << " gps " << gps_measurements[gps_counter - 1].position.transpose() << endl << endl;
//                fprintf(fp_pose, "%f,%f,%f\n", tvec.x(), tvec.y(), tvec.z());

                    auto front_idx_gps = gps_sw[0];
                    Pose3 pose_gps = values.at<Pose3>(X(front_idx));
                    fprintf(fp_pose, "%f,%f,%f\n", pose_gps.x(), pose_gps.y(), pose_gps.z());
//                    cout << "----------" << frame_counter << endl;


                    values.erase(X(front_idx));
                    values.erase(V(front_idx));
                    values.erase(B(front_idx));

                    valid_x_idx.erase(gps_sw[0]);//从所有现有节点里面滑出去
                    gps_sw.erase(gps_sw.begin());//从gps窗里面滑出去
                    if(!incorrect_angle)
                        processImg = true;
                    incorrect_angle = false;

//                first_frame = true;
//                initialized = false;

                } else {

                    while (img_measurements[img_counter].time < t_previous + tag) //找到在GPS之后最近的一帧图像
                        img_counter++;

                    //添加预积分项
                    auto img_time = img_measurements[img_counter].time;
                    current_bias = values.at<imuBias::ConstantBias>(B(frame_counter - 1));
                    current_summarized_measurement = std::make_shared<PreintegratedImuMeasurements>(imu_params,
                                                                                                    current_bias);
                    static size_t included_imu_measurement_count = 0;
                    double total_Dt = 0;
                    while (imu_counter < imu_measurements.size() && imu_measurements[imu_counter].time <= img_time) {
                        if (imu_measurements[imu_counter].time >= t_previous) {
                            if (imu_measurements[imu_counter].dt > 0) {
                                current_summarized_measurement->integrateMeasurement(
                                        imu_measurements[imu_counter].accelerometer,
                                        imu_measurements[imu_counter].gyroscope,
                                        imu_measurements[imu_counter].dt);
                                total_Dt += imu_measurements[imu_counter].dt;
                                included_imu_measurement_count++;
                            } else
                                cout << "dt < 0 ----------- " << imu_measurements[imu_counter].dt << endl;
                        }
                        imu_counter++;
                    }
                    if(imu_measurements[imu_counter].time - img_time > 0.1)
                        total_Dt = 0.5;

                    Pose3 img_pose_c;

                    included_imu_measurement_count = 1;

                    if(total_Dt < 0.3) { //只有积分是对的才加他
                        graph.emplace_shared<ImuFactor>(X(frame_counter - 1), V(frame_counter - 1),
                                                        X(frame_counter), V(frame_counter),
                                                        B(frame_counter - 1), *current_summarized_measurement);
                        auto sigma_between_b = noiseModel::Diagonal::Sigmas((Vector6() <<
                                                                                       Vector3::Constant(
                                                                                               sqrt(included_imu_measurement_count) *
                                                                                               kitti_calibration.accelerometer_bias_sigma),
                                Vector3::Constant(
                                        sqrt(included_imu_measurement_count) * kitti_calibration.gyroscope_bias_sigma))
                                                                                    .finished());
                        graph.emplace_shared<BetweenFactor<imuBias::ConstantBias>>(B(frame_counter - 1),
                                                                                   B(frame_counter),
                                                                                   imuBias::ConstantBias(),
                                                                                   sigma_between_b);
                        NavState predict;
                        predict = current_summarized_measurement->predict(
                                NavState(values.at(X(frame_counter - 1)).cast<Pose3>(),
                                         values.at(V(frame_counter - 1)).cast<Velocity3>()),
                                values.at(
                                        B(frame_counter - 1)).cast<imuBias::ConstantBias>());
                        auto img_pose = Pose3(Rot3(predict.R()), predict.t());
                        img_pose_c = img_pose.compose(C2I);

                        //如果有IMU就靠IMU往前推 如果没有IMU就单纯靠视觉往前推就行
                        values.insert(X(frame_counter), img_pose);
                        values.insert(V(frame_counter), predict.velocity());
                        values.insert(B(frame_counter), current_bias);
                    }
                    //添加投影项
                    auto img_p = img_measurements[img_counter].path;
                    cv::Mat img = cv::imread(img_p, cv::IMREAD_GRAYSCALE);
                    tracker->readImage(img);
                    img_frame_id2global_frame_id[tracker->frame_id - 1] = frame_counter;
                    img_sw.emplace_back(frame_counter);

                    //TODO 在这里新建一个东西来验证PNP求解的结果和IMU推出来的是否相符
                    std::vector<cv::Point2f> imagePoints;
                    std::vector<cv::Point3f> objectPoints;

                    auto cur_frame = tracker->all_frames.end() - 1;
                    for (int i = 0; i < cur_frame->undistort_pts.size(); i++) {
                        auto track_id = cur_frame->pt2track[i];
                        //如果track 已经建立 直接加入projection factor
                        if (valid_l_idx.count(track_id) > 0) {

                            imagePoints.push_back(
                                    cv::Point2f(cur_frame->undistort_pts[i].x, cur_frame->undistort_pts[i].y));
                            auto pt3 = values.at<Point3>(L(track_id));
                            objectPoints.emplace_back(cv::Point3f(pt3.x(), pt3.y(), pt3.z()));
                        }
                    }
                    cv::Mat rvec_cv = cv::Mat_<float>(3,1);
                    cv::Mat tvec_cv = cv::Mat_<float>(3,1);
                    cv::Mat inliers;
                    cout << "SW process imu" <<endl;
                    cout << "yyyyyyyyyyyyyyyyyyyyy solve pnp: "<< objectPoints.size() << endl;

                    Matrix4d cam_pose;
                    if(objectPoints.size() > 20) {
                        cv::solvePnPRansac(objectPoints, imagePoints, cam_K, cv::Mat(),
                                           rvec_cv, tvec_cv, false, 3000, 4, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

                        Eigen::Isometry3d MAT(Eigen::Matrix4d::Identity());
                        Eigen::Vector3d EigenTvec;
                        Eigen::Vector3d EigenRvec;
                        cv::cv2eigen(rvec_cv, EigenRvec);
                        cv::cv2eigen(tvec_cv, EigenTvec);
                        Eigen::AngleAxisd rote(EigenRvec.norm(), EigenRvec.normalized());
                        MAT.matrix().block<3, 3>(0, 0) = rote.matrix();
                        MAT.matrix().block<3, 1>(0, 3) = EigenTvec;
                        cam_pose = MAT.matrix().inverse();
                    } else
                    {
                        cerr << "not solve pnp!!!!!" << endl;
                        cam_pose = Pose3(values.at<Pose3>(X(frame_counter-1)).matrix()).compose(C2I).matrix();
                    }

                    visual_only = false;
                    if(total_Dt >= 0.3)
                    {
                        cout << endl;
                        cout << "only vision ----------------------------------" << endl;
                        cout << "only vision ----------------------------------" << endl;
                        cout << "only vision ----------------------------------" << endl;
                        cout << "only vision ----------------------------------" << endl;
                        cout << endl;
                        visual_only = true;
                        img_pose_c = Pose3(cam_pose);
                        auto img_pose = img_pose_c.compose(C2I.inverse());
                        //如果有IMU就靠IMU往前推 如果没有IMU就单纯靠视觉往前推就行
                        values.insert(X(frame_counter), img_pose);
                        values.insert(V(frame_counter), values.at(V(frame_counter-1)));
                        values.insert(B(frame_counter), values.at(B(frame_counter-1)));

                        auto velo_prior_noise = noiseModel::Isotropic::Sigma(3, 10);
                        auto bias_prior_noise = noiseModel::Isotropic::Sigma(6, 10);
                        auto velo_prior = values.at<Velocity3>(V(frame_counter-1));
                        auto bias_prior = values.at<imuBias::ConstantBias>(B(frame_counter-1));
                        graph.addPrior(V(frame_counter), velo_prior, velo_prior_noise);  // add directly to graph
                        graph.addPrior(B(frame_counter), bias_prior, bias_prior_noise);  // add directly to graph
                    }

                    vector<cv::Point2f> initialized_pts,uninitialized_pts,marged_pts;
                    vector<string> track_len;

                    cout << "IMU PREDICT: "<<img_pose_c.translation().transpose() << " " << img_pose_c.compose(C2I.inverse()).translation().transpose() << endl;
                    cout << "PNP PREDICT: "<<cam_pose.block<3, 1>(0, 3).transpose() << endl;
                    cout << "Err between: "<<(img_pose_c.translation().transpose()-cam_pose.block<3, 1>(0, 3).transpose()).norm() << endl;
                    cout << "total dt: "<<total_Dt << " size: " <<cur_frame->undistort_pts.size() << endl;

                    for (int i = 0; i < cur_frame->undistort_pts.size(); i++) {
                        auto track_id = cur_frame->pt2track[i];
                        //如果track 已经建立 直接加入projection factor
                        if (valid_l_idx.count(track_id) > 0) {

                            auto pt3 = values.at<Point3>(L(track_id));
                            Pose3 imu_pose = values.at(X(frame_counter)).cast<Pose3>();
                            Pose3 cam_pose = imu_pose.compose(C2I);
                            Point3 pt_local = cam_pose.transformTo(pt3);

                            if (pt_local.z() < 0 || pt_local.z() > 150) {
                                continue;
                            }

                            initialized_pts.emplace_back(cur_frame->distort_pts[i]);

                            if(visual_only)//todo 如果退化到纯视觉 给特征点加一个强先验来保证尺度不发散
                            {
                                auto pointNoise = noiseModel::Isotropic::Sigma(3, 0.5);
                                graph.addPrior(L(track_id), pt3, pointNoise);  // add directly to graph
                            }


                            graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
                                    Point2(cur_frame->undistort_pts[i].x, cur_frame->undistort_pts[i].y),
                                    noise_model_pt2d_robust, X(frame_counter), L(track_id),
                                    cam_K_gtsam,
                                    C2I);

                        } else if (tracker->trackid2idx.count(track_id) <= 0) //如果已经没有这个点
                        {
                            marged_pts.emplace_back(cur_frame->distort_pts[i]);
                            continue;
                        }

                        //如果track 还没有建立但已经满足建立条件
                        else {

                            uninitialized_pts.emplace_back(cur_frame->distort_pts[i]);
                            stringstream ss;
                            ss << (tracker->all_tracks[tracker->trackid2idx[track_id]].active ? 'a':'b');
                            ss << tracker->all_tracks[tracker->trackid2idx[track_id]].imgIdAndPtID.size();

                            track_len.emplace_back(ss.str());

                            if (!tracker->all_tracks[tracker->trackid2idx[track_id]].active &&
                                tracker->all_tracks[tracker->trackid2idx[track_id]].imgIdAndPtID.size() > 3) {
                                auto track = tracker->all_tracks[tracker->trackid2idx[track_id]];
                                std::vector<Pose3> poses;
                                Point2Vector measurements;
                                vector<int> poseIdx;
                                for (auto pair: track.imgIdAndPtID) {
                                    int frame_id = pair.first, pt_id = pair.second;
                                    for (auto j: tracker->all_frames) {
                                        if (j.frame_id == frame_id) {
                                            Pose3 imu_pose = values.at(
                                                    X(img_frame_id2global_frame_id[frame_id])).cast<Pose3>();
                                            Pose3 cam_pose = imu_pose.compose(C2I);
                                            poses.push_back(cam_pose);
                                            measurements.push_back(
                                                    Point2(j.undistort_pts[pt_id].x, j.undistort_pts[pt_id].y));
                                            poseIdx.push_back(img_frame_id2global_frame_id[frame_id]);
                                        }
                                    }
                                }

                                if (poses.size() > 2) {

                                    auto pt3d = myTriangulatePoint3(poses, cam_K_gtsam, measurements);
                                    Pose3 imu_pose = values.at(X(poseIdx[0])).cast<Pose3>();
                                    Pose3 cam_pose = imu_pose.compose(C2I);
                                    Point3 pt_local = cam_pose.transformTo(pt3d);

                                    if (pt_local.z() < 0 || pt_local.z() > 150) {
                                        auto dis = values.at(X(poseIdx[1])).cast<Pose3>().compose(C2I).translation() - cam_pose.translation();
                                        auto axis = (cam_pose.rotation().transpose() * dis).normalized().transpose();
                                        if(axis.z()<0.8)
                                        {
                                            cout << "!!!!!!!!!!!!!!!!!" << "changed" << endl;
                                            cout << axis.transpose() << endl;
//                                            incorrect_angle = true;
//                                            gps_counter -= 20;
//                                            img_counter -= 20;
//                                            img_counter -= 2000;
//                                            first_frame = true;
//                                            initialized = false;
//                                            break;
                                        }


//                                        cout << dis.transpose() << endl;
//                                        cout << (cam_pose.rotation().transpose() * dis).normalized().transpose() << endl;
                                        continue;
                                    }

                                    track.active = true;

                                    auto pointNoise = noiseModel::Isotropic::Sigma(3, 10);
                                    graph.addPrior(L(track.id), pt3d, pointNoise);  // add directly to graph

                                    values.insert(L(track.id), pt3d);
                                    valid_l_idx[track.id] = 0;
                                    for (int i = 0; i < poseIdx.size(); i++) {
                                        Pose3 imu_pose = values.at(X(poseIdx[i])).cast<Pose3>();
                                        Pose3 cam_pose = imu_pose.compose(C2I);
                                        Point3 pt_local = cam_pose.transformTo(pt3d);
                                        if (pt_local.z() < 0)
                                            continue;
                                        graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
                                                measurements[i], noise_model_pt2d_robust, X(poseIdx[i]), L(track.id),
                                                cam_K_gtsam,
                                                C2I);
                                    }
                                }
                                else
                                {
//                                    cout << "less pose" <<endl;
                                    int a= 1;
                                }
                            }
                        }
                    }

                    showTracking(cur_frame->img,initialized_pts,uninitialized_pts,track_len,marged_pts);


                    //求解 边缘化
                    LevenbergMarquardtParams params;
                    LevenbergMarquardtParams::SetCeresDefaults(&params);
                    cout << "before: "<< values.at<Pose3>(X(frame_counter)).rotation() << endl;
                    values = LevenbergMarquardtOptimizer(graph, values, params).optimize();

                    cout << "OPTIMIZE: " << values.at<Pose3>(X(frame_counter)).translation().transpose() << endl;
                    cout << "after: "<< values.at<Pose3>(X(frame_counter)).rotation() << endl;

                    KeyVector marg_keys;
                    int front_idx = img_sw[0];

                    //还有要被marg掉的路标点
                    marg_keys.push_back(X(front_idx));
                    if(values.exists(V(front_idx)) > 0) {
                        marg_keys.push_back(V(front_idx));
                        marg_keys.push_back(B(front_idx));
                    }

                    //TODO 路标点的marg还要好好看看 不能直接跟着关键帧就扔了 需要被扔掉的是那种在窗里面已经不被观测
                    auto all_keys = graph.keys();
                    auto frame = tracker->all_frames[0];
                    for (auto i: frame.host_track) {
                        if (valid_l_idx.count(i) > 0) {
                            if (all_keys.find(L(i)) != all_keys.end())
                                marg_keys.push_back(L(i));
                        }
                    }


                    auto linear_graph = graph.linearize(values);
                    try {
                        auto marginalLinearFactors = BatchFixedLagSmoother::CalculateMarginalFactors(*linear_graph,
                                                                                                     marg_keys,
                                                                                                     gtsam::EliminateQR);
                        graph = LinearContainerFactor::ConvertLinearGraph(marginalLinearFactors, values);
                    }
                    catch (exception& e) {
                        cout << "Standard exception: " << e.what()  << endl;
                    }


//                    Pose3 pose = values.at<Pose3>(X(frame_counter));
                    Pose3 pose = values.at<Pose3>(X(front_idx));
                    Pose3 img_pose = pose.compose(C2I);
                    auto tvec = pose.translation();
//                    cout << tvec << endl;
                    fprintf(fp_pose, "%f,%f,%f\n", tvec.x(), tvec.y(), tvec.z());
                    cout << "----------" << frame_counter << endl << endl;

                    for (auto i: frame.host_track) {
                        if (valid_l_idx.count(i) > 0) {
                            auto pt3 = values.at<Point3>(L(i));
                            auto pt_local = img_pose.transformTo(pt3);
                            if(pt_local.z() > 100 || pt_local.z() < 0)
                                continue;
                            fprintf(fp_pts, "%f,%f,%f\n", pt3.x(), pt3.y(), pt3.z());
                        }
                    }

                    values.erase(X(front_idx));
                    values.erase(V(front_idx));
                    values.erase(B(front_idx));

                    for (auto i: frame.host_track) {
                        if (valid_l_idx.count(i) > 0) {
                            values.erase(L(i));
                            valid_l_idx.erase(i);
                        }
                    }

                    valid_x_idx.erase(img_sw[0]);//从所有现有节点里面滑出去
                    img_sw.erase(img_sw.begin());//从gps窗里面滑出去
                    //还有tracker 里面的路标点和frame需要处理 这个先不删除

                    for (auto i: frame.host_track) {
                        if (tracker->trackid2idx.count(i) > 0) {
                            tracker->all_tracks[tracker->trackid2idx[i]].active = true;
                        }
                    }

                    tracker->all_frames.erase(tracker->all_frames.begin());
                    frame_counter++;
                    img_counter++;
                    t_previous = img_time;
                    processImg = false;
                    cout << " step new " << endl;
                }
                cout << 1 << endl;
            }
//            cout << 2 << endl;
            catch(...)
            {
                cout << "reboot: " << frame_counter << endl;
                first_frame = true;
                initialized = false;
            }
        }
        cout << 3 << endl;
    }

    return 0;
}