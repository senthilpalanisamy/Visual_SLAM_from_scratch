#include<stereo_slam/slam_utilities.hpp>

Camera::Camera(double fx_, double fy_, double cx_, double cy_, SE3 pose):fx(fx_), fy(fy_), cx(cx_), cy(cy_), pose(pose)
{

};

Mat33 Camera::K()
{
  Mat33 k;
  k << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
  return k;
}

Vec2 Camera::cameraToPixels(const Vec3& point3d)
{
  return Vec2(fx * point3d(0, 0) / point3d(2, 0) + cx,
             fy * point3d(1, 0) / point3d(2, 0) + cy);
}

Vec3 Camera::pixelToCamera(const Vec2& pixel, double depth)
{
  return Vec3((pixel(0, 0) - cx) / fx * depth,
               (pixel(1, 0) - cy) / fy * depth,
               depth);
}
