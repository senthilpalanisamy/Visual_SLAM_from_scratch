#ifndef VISUAL_ODOMETRY_INCLUDE_HPP
#define VISUAL_ODOMETRY_INCLUDE_HPP

#include <opencv2/opencv.hpp>
#include "stereo_slam/frontend.hpp"
#include "stereo_slam/dataset.hpp"
#include "stereo_slam/common_include.hpp"


class SVO
{

  TrackingState trackingState;
  FrontEnd frontend;
  KittiDatasetAdapter dataset;
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  SVO(const KittiDatasetAdapter& dataset_):dataset(dataset_){};
  void run();
  void init();

};
#endif
