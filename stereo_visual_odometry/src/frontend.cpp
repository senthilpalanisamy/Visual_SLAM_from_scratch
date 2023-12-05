#include <stereo_slam/frontend.hpp>
#include <stereo_slam/slam_utilities.hpp>
#include <stereo_slam/algorithms.hpp>

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
    if(stereoInit())
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
  cv::imshow("leftImage", currentFrame->leftImage);
  cv::imshow("rightImage", currentFrame->rightImage);
  cv::waitKey(100);

  std::cout<<"type: "<<currentFrame->leftImage.type()<<std::endl;
  std::cout<<"type: "<<currentFrame->rightImage.type()<<std::endl;
  std::cout<<"channels: "<<currentFrame->leftImage.channels()<<std::endl;
  std::cout<<"channels: "<<currentFrame->rightImage.channels()<<std::endl;
  cv::calcOpticalFlowPyrLK(currentFrame->leftImage, currentFrame->rightImage,
      left, right, status, error, cv::Size(11, 11), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
  int goodFeaturesCount=0;
  for(int i=0; i < status.size(); ++i)
  {
    if(status[i])
    {
      cv::KeyPoint rightKeyPoint(right[i], 7);
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


  // visualise
  cv::Mat leftImageColor, rightImageColor;
  cv::cvtColor(currentFrame->leftImage, leftImageColor, cv::COLOR_GRAY2RGB);
  cv::cvtColor(currentFrame->rightImage, rightImageColor, cv::COLOR_GRAY2RGB);
  vector<cv::DMatch> matches;
  for(int i=0; i < status.size(); ++i)
  {
    if(status[i])
    {
      matches.emplace_back(i, i, 0);
    }
  }
  vector<cv::KeyPoint> leftKeyPoint, rightKeyPoint;
  for(auto pt:left)
  {
    leftKeyPoint.emplace_back(pt, 7.0);
  }
  for(auto pt:right)
  {
    rightKeyPoint.emplace_back(pt, 7.0);
  }
  cv::Mat output;
  cv::drawMatches(leftImageColor, leftKeyPoint, rightImageColor, rightKeyPoint, matches, output);
  cv::imshow("matches", output);
  cv::waitKey(100);
  std::cout<<"exiting"<<std::endl;
  return goodFeaturesCount;

}

void FrontEnd::setCameras(Camera::Ptr leftCamera_, Camera::Ptr rightCamera_)
{
  leftCamera = leftCamera_;
  rightCamera = rightCamera_;
}


bool FrontEnd::stereoInit()
{
  int leftFeatures = detectFeatures();
  int rightFeatures = findFeaturesInRight();
  int featuresThreshold = 30;
  if(rightFeatures < featuresThreshold)
  {
    return false;

  }
  bool isMapInit = mapInit();
  return isMapInit;
}

bool FrontEnd::mapInit()
{

  for(int i=0; i < currentFrame->leftFeatures.size(); ++i)
  {
    if((currentFrame->leftFeatures[i] == nullptr) || (currentFrame->rightFeatures[i] == nullptr))
    {
      continue;
    }

    // change intrinsics
    vector<Vec3> points {leftCamera->pixelToCamera(Vec2(currentFrame->leftFeatures[i]->kp.pt.x,
                                                     currentFrame->rightFeatures[i]->kp.pt.y)),
                          rightCamera->pixelToCamera(Vec2(currentFrame->rightFeatures[i]->kp.pt.x,
                                                       currentFrame->rightFeatures[i]->kp.pt.y))};
    vector<SE3> poses = {leftCamera->pose, rightCamera->pose};
    Vec3 pWorld = Vec3::Zero();
    if(!triangulate(poses,points, pWorld) && (pWorld[2] > 0))
    {
      std::cout<<"here"<<std::endl;

    }
  }


  return false;

}
