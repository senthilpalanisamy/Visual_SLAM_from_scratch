#include<stereo_slam/dataset.hpp>
#include<stereo_slam/visual_odometry.hpp>

int main()
{
  string kittiPath = "/home/senthil/work/slam-construction-from-scratch/data_odometry_gray/dataset/sequences/00";
  KittiDatasetAdapter kittiData(kittiPath);
  SVO svo(kittiData);
  svo.init();
  svo.run();
  

  return 0;
}
