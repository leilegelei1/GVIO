//
// Created by jerry on 2021/9/19.
//

#ifndef GVIO_FEATURE_TRACKER_H
#define GVIO_FEATURE_TRACKER_H

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <vector>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <baseStruct.h>

using namespace std;
using namespace Eigen;


bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker {
public:
    FeatureTracker(bool withMarginal=false);

    bool readImage(const cv::Mat &_img);

    void setMask();
    void setMask_simple();

    void addPoints();

    void rejectWithF();

    void undistortedPoints();

    int checkParallaxPtPair(cv::Point2f pt_i,cv::Point2d pt_j);
    bool checkParallaxImagePair();

    cv::Mat mask;
    cv::Mat cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> cur_pts, forw_pts;
    vector<cv::Point2f> cur_un_pts;
    vector<int> ids;
    vector<int> track_cnt;

    vector<int> cur_pt_ids;

    vector<Track> all_tracks;
    vector<Frame> all_frames;
    vector<int> all_frames_id;

    unordered_map<int,int> trackid2idx;//记录从trackid 到 对应all_tracks 索引的图
    unordered_map<int,int> framdid2idx;//记录从frameid 到 对应all_frame 索引的图

    void update_trackid2idx();

    int frame_id = 0;
    int track_id = 0;

    bool withMarginal_ = true;
    int trackUpdateTag = 0;
};

#endif //GVIO_FEATURE_TRACKER_H
