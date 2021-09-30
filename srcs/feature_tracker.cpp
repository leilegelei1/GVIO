//
// Created by jerry on 2021/9/19.
//

#include "feature_tracker.h"
#include <parameters.h>
#include <cmath>

bool inBorder(const cv::Point2f &pt)
{

    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{

    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker(bool withMarginal)
{
    withMarginal_ = withMarginal;
}

void FeatureTracker::setMask_simple()
{
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    ids.clear();
    track_cnt.clear();
    for(auto pt:forw_pts)
    {
        ids.push_back(-1);
        track_cnt.push_back(-1);
        cv::circle(mask, pt, MIN_DIST, 0, -1);
    }
}

void FeatureTracker::setMask()
{
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
    {
        return a.first > b.first;
    });

    forw_pts.clear(); // 应该是这里坏事了 把整体的数据全弄偏了
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::update_trackid2idx()
{
//    trackid2idx.clear();
//    cout << "all tracks size: " << all_tracks.size() << " all frames size: " << all_frames.size() << endl;
//    for(int i=0;i<all_tracks.size();i++)
//    {
//        trackid2idx[all_tracks[i].id] = i;
//    }

    for(;trackUpdateTag<all_tracks.size();trackUpdateTag++)
    {
        trackid2idx[all_tracks[trackUpdateTag].id] = trackUpdateTag;
    }

    set<int> frame_ids;
    framdid2idx.clear();
    for(int i=0;i<all_frames.size();++i)
    {
        framdid2idx[all_frames[i].frame_id] = i;
        frame_ids.insert(all_frames[i].frame_id);
    }
}

bool FeatureTracker::readImage(const cv::Mat &_img)//同时返回对track情况的判断
{
    cv::Mat img;
    img = _img;

    Frame new_frame;//新建一个frame
//    Frame cur_frame;
    Frame cur_frame;
    bool marg_type = true;

    //如果是第一帧那就是个起始帧
    if (forw_img.empty())
    {
        cur_img = forw_img = img;
    }
    else //第二帧才开始真正的初始化
    {
        forw_img = img;
        cur_frame = all_frames[all_frames.size()-1];
        cur_img = cur_frame.img;
        cur_pts = cur_frame.distort_pts;

    }

    forw_pts.clear();
    cur_pt_ids.clear();
    for(int i=0;i<cur_pts.size();i++)
        cur_pt_ids.push_back(i);

    vector<int> host_track;
    if (cur_pts.size() > 0)
    {
        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;

        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        reduceVector(cur_pt_ids,status);

        for (auto &n : track_cnt)
            n++;

        rejectWithF();

        marg_type = checkParallaxImagePair() > MIN_PARALLAX; //考虑是否需要把他marg掉了

        for(int i=0;i<cur_pt_ids.size();i++)
        {
            int left=cur_pt_ids[i],right=i;
            int now_track_id = cur_frame.pt2track[left];

            auto first_frame_ix = all_tracks[now_track_id].imgIdAndPtID[0].first;
            if(withMarginal_) {
                if (framdid2idx.count(first_frame_ix) > 0) {
                    all_tracks[trackid2idx[now_track_id]].imgIdAndPtID.push_back(make_pair(frame_id, right));
                    new_frame.pt2track[right] = now_track_id;
                } else //TODO 这里其实感觉挺奇怪的　明明他们就是一个track链里面的 现在因为边缘化的存在 导致我在操作的过程中把这个点给边缘化掉了 不能再进行投影 那如果不进行边缘化会怎么样呢？
                {
                    Track track;
                    track.id = track_id++;
                    track.active = false;
                    track.imgIdAndPtID.push_back(make_pair(frame_id, right));
                    all_tracks.push_back(track);
                    host_track.push_back(track.id);
                    new_frame.pt2track[i] = track.id;
                }
            }
            else
            {
                all_tracks[trackid2idx[now_track_id]].imgIdAndPtID.push_back(make_pair(frame_id, right));
                new_frame.pt2track[right] = now_track_id;
            }
        }
    }

    setMask_simple();

    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0)
    {
        if(mask.empty())
            cout << "mask is empty " << endl;
        if (mask.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        if (mask.size() != forw_img.size())
            cout << "wrong size " << endl;
        cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        //TODO 生成了n_pts得放进去
    }
    else
        n_pts.clear();

    //n_pts 得整合到forw_pts 这一步好像忘记了

    //新建该帧为主导的track
    for(int i=forw_pts.size();i<MAX_CNT;++i)
    {
        Track track;
        track.id = track_id++;
        track.active = false;
        track.imgIdAndPtID.push_back(make_pair(frame_id,i));
        all_tracks.push_back(track);
        host_track.push_back(track.id);
        new_frame.pt2track[i] = track.id;
    }

    addPoints();
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();

    //把对应的数据写进全量的buffer里面
    new_frame.frame_id = frame_id++;
    new_frame.img = forw_img;
    new_frame.distort_pts = forw_pts;
    new_frame.undistort_pts = cur_un_pts;
    new_frame.host_track = host_track;
    all_frames.push_back(new_frame);
    all_frames_id.push_back(new_frame.frame_id);

    auto t1 = std::chrono::steady_clock::now();
    update_trackid2idx(); //更新索引
    auto t2 = std::chrono::steady_clock::now();
    cout << "update time: " << std::chrono::duration<double,std::milli>(t2-t1).count() << endl;

    return marg_type;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        vector<uchar> status;
        cv::findFundamentalMat(cur_pts, forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        reduceVector(cur_pt_ids,status);
    }
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cv::undistortPoints(cur_pts, cur_un_pts, cam_K, cam_dist, cv::noArray(),cam_K);

    int a=1;
}

int FeatureTracker::checkParallaxPtPair(cv::Point2f pt_i,cv::Point2d pt_j)
{
    double du = pt_i.x - pt_j.x;
    double dv = pt_i.y - pt_j.y;
    return sqrt(du*du + dv*dv);
}
bool FeatureTracker::checkParallaxImagePair()
{
    int totalParallax = 0;
    for(int i=0;i<cur_pts.size();i++)
    {
        totalParallax+= checkParallaxPtPair(cur_pts[i],forw_pts[i]);
    }
    return totalParallax / (float)(cur_pts.size());
}
