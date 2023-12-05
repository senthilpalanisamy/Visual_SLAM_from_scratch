#ifndef COMMON_INCLUDE_HPP
#define COMMON_INCLUDE_HPP
#include<Eigen/Core>
#include<Eigen/Geometry>
#include<vector>

typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, 3, 4> Mat34;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;


// SE3
#include<sophus/se3.hpp>
#include<sophus/so3.hpp>
typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;

using std::vector;

#endif
