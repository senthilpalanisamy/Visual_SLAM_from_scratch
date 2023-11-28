#ifndef FRONT_END_INCLUDE_HPP
#define FRONT_END_INCLUDE_HPP
#include<stereo_slam/frame.hpp>
#include<stereo_slam/slam_utilities.hpp>

enum class TrackingState{INITING, TRACKING_GOOD, TRACKING_BAD, LOST};

class FrontEnd
{
  shared_ptr<Frame> currentFrame, lastFrame;
  cv::Ptr<cv::GFTTDetector> gftt;
  TrackingState trackingState;
  Camera::Ptr leftCamera, rightCamera;

  int detectFeatures();
  int findFeaturesInRight();
  bool stereoInit();
  bool mapInit();
  public:
  FrontEnd();
  void addFrame(shared_ptr<Frame>& frame);
  void setCameras(Camera::Ptr leftCamera, Camera::Ptr rightCamera);

};
#endif
