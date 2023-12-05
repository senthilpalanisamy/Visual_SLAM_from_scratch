#ifndef ALGORITHMS_INCLUDE_HPP
#define ALGORITHMS_INCLUDE_HPP
#include <stereo_slam/common_include.hpp>

bool triangulate(const vector<SE3>& poses, const vector<Vec3>& pixels, Vec3& point3d)
{
  MatXX M(poses.size() * 2, 4);
  VecX b(poses.size()* 2);
  b.setZero();

  for(int i=0; i < poses.size(); ++i)
  {
    Mat34 m = poses[i].matrix3x4();
    M.block<1,4>(2*i, 0) = pixels[i][0] * m.row(2) - m.row(0);
    M.block<1,4>(2*i+1, 0) = pixels[i][1] * m.row(2) - m.row(1);
  }
  auto svd = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  point3d = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();
  if(svd.singularValues()[3] / svd.singularValues()[2] < 1e-2)
  {
    return true;
  }
  return false;
}

#endif
