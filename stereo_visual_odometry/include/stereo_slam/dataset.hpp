#ifndef DATASET_H
#define DATASET_H
#include<string>
#include<vector>

#include<boost/format.hpp>

#include<stereo_slam/slam_utilities.hpp>
#include<stereo_slam/frame.hpp>

using std::vector;
using std::string;

class KittiDatasetAdapter
{
  string dataPath;
  const string calibFile = "calib.txt";
  const double downsamplingFactor = 0.5;
  int imageIndex;
  boost::format imagePathFormat;
  public:
  vector<Camera::Ptr> cameras;
  KittiDatasetAdapter(const string& dataPath_);
  Frame::Ptr nextFrame();
};
#endif
