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
    if(stereoInit());
    {
      trackingState = TrackingState::TRACKING_GOOD;
    }
  }

  lastFrame = currentFrame;

}

int FrontEnd::findFeaturesInRight()
{
  vector<cv::Point2f> right, left;
  for(auto& feat: currentFrame->leftFeatures)
  {
    left.push_back(feat->kp.pt);
    // add map based projection
    right.push_back(feat->kp.pt);
  }
  std::vector<uchar> status;
  cv::Mat error;
  cv::calcOpticalFlowPyrLK(currentFrame->leftImage, currentFrame->rightImage,
      left, right, error, status, cv::Size(11, 11), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
  int goodFeaturesCount=0;
  for(int i=0; i < status.size(); ++i)
  {
    if(status[i])
    {
      cv::KeyPoint rightKeyPoint(right[1], 7);
      Feature::Ptr feat(new Feature(rightKeyPoint));
      feat->isOnLeftImage = false;
      feat->isOnRightImage = true;
      currentFrame->rightFeatures.push_back(feat);
      ++goodFeaturesCount;
    }
    else
    {
      currentFrame->rightFeatures.push_back(nullptr);
    }
  }
  return goodFeaturesCount;

}


bool FrontEnd::stereoInit()
{
  int leftFeatures = detectFeatures();
  int rightFeatures = findFeaturesInRight();
  int featuresThreshold = 100;
  if(rightFeatures < featuresThreshold)
  {
    return false;

  }


}
