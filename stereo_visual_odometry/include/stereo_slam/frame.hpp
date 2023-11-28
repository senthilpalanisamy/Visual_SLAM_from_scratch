#ifndef FRAME_INCLUDE_HPP
#define FRAME_INCLUDE_HPP
#include <memory>
#include <vector>

class Feature;
class Camera;

#include <opencv2/opencv.hpp>

//#include <stereo_slam/slam_utilities.hpp>


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
