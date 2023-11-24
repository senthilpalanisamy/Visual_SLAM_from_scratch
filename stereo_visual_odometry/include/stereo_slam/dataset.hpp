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
  vector<Camera::Ptr> cameras;
  int imageIndex;
  boost::format imagePathFormat;
  public:
  KittiDatasetAdapter(const string& dataPath_);
  Frame::Ptr nextFrame();
};
#endif
