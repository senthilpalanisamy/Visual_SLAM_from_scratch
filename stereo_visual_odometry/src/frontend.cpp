#include <stereo_slam/frontend.hpp>
#include <stereo_slam/slam_utilities.hpp>

FrontEnd::FrontEnd()
{
  gftt = cv::GFTTDetector::create(200, 0.1, 20);
  trackingState = TrackingState::INITING;
}

int FrontEnd::detectFeatures()
{
  vector<cv::KeyPoint> keyPoints;
  gftt->detect(currentFrame->leftImage, keyPoints);
  for(auto& kp: keyPoints)
  {
    currentFrame->leftFeatures.push_back(Feature::Ptr(new Feature(kp, currentFrame)));
  }
  return keyPoints.size();

}

void FrontEnd::addFrame(shared_ptr<Frame>& currentFrame_)
{
  currentFrame = currentFrame_;
  if(trackingState == TrackingState::INITING)
  {
    stereoInit();
  }
  lastFrame = currentFrame;

}


void FrontEnd::stereoInit()
{
  detectFeatures();


}
