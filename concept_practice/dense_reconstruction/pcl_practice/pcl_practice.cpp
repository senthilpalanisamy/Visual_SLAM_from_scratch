#include <boost/format.hpp>
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<Eigen/Dense>


using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
  vector<Mat> colorImages, depthImages;
  int imageCount=5;
  boost::format formatString("./data/%s/%d.png");
  vector<Eigen::Isometry3d> poses;

  ifstream poseReader("./data/pose.txt");

  for(int i=1; i <= imageCount; ++i)
  {
    string colorImagePath = (formatString % "color" % i).str();
    string depthImagePath = (formatString % "depth" % i).str();

    //Mat colorImage = cv::imread(colorImagePath);
    //Mat depthImage = cv::imread(depthImagePath, cv::IMREAD_UNCHANGED);

    colorImages.emplace_back(imread(colorImagePath));
    depthImages.emplace_back(imread(depthImagePath, cv::IMREAD_UNCHANGED));


    // vector<double> currentPose(7, 0);
    double currentPose[7] = {0.0};
    for(int k=0; k < 7; ++k)
    {
      poseReader >> currentPose[k];
    }

    Eigen::Quaterniond q(currentPose[6], currentPose[3], currentPose[4], currentPose[5]);
    Eigen::Isometry3d T(q);
    T.pretranslate(Eigen::Vector3d(currentPose[0], currentPose[1], currentPose[2]));
    poses.emplace_back(T);
  }

  double cx = 319.5;
  double cy = 239.5;
  double fx = 481.2;
  double fy = -480.0;
  double depthScale = 5000.0;

  typedef pcl::PointXYZRGB PointT;
  typedef pcl::PointCloud<PointT> PointCloud;

  PointCloud::Ptr pointcloud(new PointCloud);
  int width = colorImages[0].cols;
  int height = colorImages[0].rows;

  for(int i=0; i <imageCount; ++i)
  {
    PointCloud::Ptr current(new PointCloud);

    for(int u=0; u < height; ++u)
    {
      for(int v=0; v < width; ++v)
      {
        double depth = depthImages[i].ptr<unsigned short>(u)[v];
        if(depth == 0)
        {
          continue;
        }

        Eigen::Vector3d point;
        point[1] = (u - cy) / fy  * static_cast<double>(depth) / depthScale;
        point[0] = (v - cx) / fx  * static_cast<double>(depth) / depthScale;
        point[2] = static_cast<double>(depth) / depthScale;

        auto point_G = poses[i] * point;
        PointT pclPoint;
        pclPoint.x = point_G[0];
        pclPoint.y = point_G[1];
        pclPoint.z = point_G[2];
        pclPoint.b = colorImages[i].ptr<cv::Vec3b>(u)[v][0];
        pclPoint.g = colorImages[i].ptr<cv::Vec3b>(u)[v][1];
        pclPoint.r = colorImages[i].ptr<cv::Vec3b>(u)[v][2];

        current->points.push_back(pclPoint);
      }

    }

    PointCloud::Ptr tmp(new PointCloud);
    pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    statistical_filter.setMeanK(50);
    statistical_filter.setStddevMulThresh(0.1);
    statistical_filter.setInputCloud(current);
    statistical_filter.filter(*tmp);
    (*pointcloud) += (*tmp);
  }

  pointcloud->is_dense=false;
  cout<<"point cloud size before voxel filtering: "<<pointcloud->size()<<endl;

  pcl::VoxelGrid<PointT> voxel_filter; 
  double resolution = 0.03;
  voxel_filter.setLeafSize(resolution, resolution, resolution);
  PointCloud::Ptr finalPointCloud(new PointCloud);
  voxel_filter.setInputCloud(pointcloud);
  voxel_filter.filter(*finalPointCloud);
  finalPointCloud->swap(*pointcloud);

  cout<<"point cloud size after voxel filtering: "<<pointcloud->size()<<endl;
  pcl::io::savePCDFileBinary("pointcloud.pcd", *pointcloud);
  return 0;
}
