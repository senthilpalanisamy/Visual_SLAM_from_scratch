#ifndef FRONT_END_INCLUDE_HPP
#define FRONT_END_INCLUDE_HPP
#include<stereo_slam/frame.hpp>

enum class TrackingState{INITING, TRACKING_GOOD, TRACKING_BAD, LOST};

class FrontEnd
{
  shared_ptr<Frame> currentFrame, lastFrame;
  cv::Ptr<cv::GFTTDetector> gftt;
  TrackingState trackingState;

  int detectFeatures();
  void stereoInit();
  public:
  FrontEnd();
  void addFrame(shared_ptr<Frame>& frame);

};
#endif
