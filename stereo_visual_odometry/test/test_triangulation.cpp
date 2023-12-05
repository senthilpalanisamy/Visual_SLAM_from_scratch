#include <gtest/gtest.h>

#include <stereo_slam/algorithms.hpp>
#include <stereo_slam/common_include.hpp>

using std::vector;

TEST(stereoslamTest, Triangulation)
{
  Vec3 point3D(20, 5, 5);
  auto T21 = SE3(Mat33::Identity(), Vec3(7, 0, 0));
  auto T11 = SE3(Mat33::Identity(), Vec3(0, 0, 0));
  Vec3 pixel1 = point3D;
  pixel1 = pixel1 / pixel1(2, 0);
  auto P2 = T21.inverse() * point3D;
  Vec3 pixel2 = P2 / P2(2, 0);
  vector<SE3> poses = {T11, T21.inverse()};
  vector<Vec3> pixels = {pixel1, pixel2};
  Vec3 point;
  EXPECT_TRUE(triangulate(poses, pixels, point));
  EXPECT_NEAR(point(0,0), point3D(0,0), 0.01);
  EXPECT_NEAR(point(1,0), point3D(1,0), 0.01);
  EXPECT_NEAR(point(2,0), point3D(2,0), 0.01);
}


int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
