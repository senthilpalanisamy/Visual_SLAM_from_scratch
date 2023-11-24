#ifndef SLAM_UTILITIES_H
#define SLAM_UTILITES_H
#include <memory>
#include <vector>

#include<opencv2/opencv.hpp>

#include <stereo_slam/frame.hpp>
using std::vector;
using std::shared_ptr;
using std::weak_ptr;


class Camera
{
  double fx, fy, cx, cy, baseline;
  public:
  typedef shared_ptr<Camera> Ptr;
  vector<Ptr> cameras;
  Camera(double fx_, double fy_, double cx_, double cy_);

};

class MapPoint
{

};

class Feature
{
  cv::KeyPoint position;
  weak_ptr<Frame> frame;
  weak_ptr<MapPoint> mapPoint;
  public:
  Feature(cv::KeyPoint position_):position(position_){};
  Feature(cv::KeyPoint position_, shared_ptr<Frame> frame_): position(position_), frame(frame_){};
  typedef shared_ptr<Feature> Ptr;
};

#endif
