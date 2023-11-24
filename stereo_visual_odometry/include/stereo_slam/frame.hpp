#ifndef FRAME_INCLUDE_HPP
#define FRAME_INCLUDE_HPP
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
class Feature;
//#include <stereo_slam/slam_utilites.hpp>

using std::shared_ptr;
using std::vector;

class Frame
{
  public:
  cv::Mat leftImage, rightImage;
  vector<shared_ptr<Feature>> leftFeatures;
  typedef shared_ptr<Frame> Ptr;
  vector<shared_ptr<Feature>> rightFeatures;
  static Frame::Ptr createFrame();

};
#endif
