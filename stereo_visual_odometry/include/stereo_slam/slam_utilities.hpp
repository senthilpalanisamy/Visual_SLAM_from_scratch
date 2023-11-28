#ifndef SLAM_UTILITIES_H
#define SLAM_UTILITIES_H
#include <memory>
#include <vector>

#include<opencv2/opencv.hpp>

#include <stereo_slam/common_include.hpp>
#include <stereo_slam/frame.hpp>

using std::vector;
using std::shared_ptr;
using std::weak_ptr;


class Camera
{
  double fx, fy, cx, cy, baseline;
  public:
  typedef shared_ptr<Camera> Ptr;
  SE3 pose;
  // typdef shared_ptr<Camera> Ptr;
  Camera(double fx_, double fy_, double cx_, double cy_, SE3 pose);
  Mat33 K();
  Vec2 cameraToPixels(const Vec3& point3d);
  Vec3 pixelToCamera(const Vec2& pixel, double depth=1.0);
};

class MapPoint
{

};

class Feature
{
  weak_ptr<Frame> frame;
  weak_ptr<MapPoint> mapPoint;
  public:
  Feature(cv::KeyPoint kp_):kp(kp_){};
  Feature(cv::KeyPoint kp_, shared_ptr<Frame> frame_): kp(kp_), frame(frame_){};
  cv::KeyPoint kp;
  typedef shared_ptr<Feature> Ptr;
  bool isOnLeftImage, isOnRightImage;
};

#endif
