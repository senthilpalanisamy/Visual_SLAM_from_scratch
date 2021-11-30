#include<opencv2/opencv.hpp>
#include "feature_matching.hpp"

using namespace std;


void find_R_and_t(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& intrinsics,
                  cv::Mat& R, cv::Mat& t, vector<cv::DMatch>& matches,
                  vector<cv::KeyPoint>& kp1, vector<cv::KeyPoint>& kp2)
{

  match_image_features(img1, img2, matches, kp1, kp2);

  vector<cv::Point2f> points1, points2;

  for(int i=0; i < matches.size(); ++i)
  {
    points1.emplace_back(kp1[matches[i].queryIdx].pt);
    points2.emplace_back(kp2[matches[i].trainIdx].pt);
  }

  cv::Mat fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
  cout<<"fundmental matrix is "<<endl<<fundamentalMatrix<<endl;

  cv::Point2d principalPoint(intrinsics.at<double>(0,2), intrinsics.at<double>(1,2));
  double focalLength = intrinsics.at<double>(1,1); // assumes both focal lenghts are equal. Revisit later

  cv::Mat essentialMatrix = cv::findEssentialMat(points1, points2, focalLength, principalPoint);
  cout<<"essential matrix is "<<endl<<essentialMatrix<<endl;

  cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3);
  cout<<"homography matrix is: "<<endl<<homography<<endl;

  cv::recoverPose(essentialMatrix, points1, points2, R, t, focalLength, principalPoint);
  cout<<"R is "<<endl<<R<<endl;
  cout<<"t is "<<endl<<t<<endl;

}

cv::Point2d pixel2Cam(const cv::Point2d& point, const cv::Mat& K)
{
  cv::Point2d normalisedPoint;
  double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), px = K.at<double>(0,2),
         py = K.at<double>(1, 2);
  normalisedPoint.x = (point.x - px) / fx;
  normalisedPoint.y = (point.y - py) / fy;
  return normalisedPoint;
}

void triangulate(const vector<cv::KeyPoint>& kp1, const vector<cv::KeyPoint>& kp2,
                 const vector<cv::DMatch> &matches, const cv::Mat &R, const cv::Mat &t,
                 cv::Mat &K, vector<cv::Point3d>& points3D)
{
  cv::Mat T1 = (cv::Mat_<double>(3, 4)<<
           1.0, 0, 0, 0,
           0, 1.0, 0, 0,
           0, 0, 1.0, 0);
  cv::Mat T2 = (cv::Mat_<double>(3, 4)<<
           R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
           R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
           R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
  cout<<"T value"<<T2.at<double>(0, 3)<<endl<<T2.at<double>(1, 3)<<endl<<T2.at<double>(2, 3)<<endl<<T2.at<double>(0,0)<<endl;
  vector<cv::Point2d> points1, points2;

  for(auto& match: matches)
  {
    auto normalised_point1 = pixel2Cam(kp1[match.queryIdx].pt, K);
    auto normalised_point2 = pixel2Cam(kp2[match.trainIdx].pt, K);

    points1.emplace_back(normalised_point1.x, normalised_point1.y);
    points2.emplace_back(normalised_point2.x, normalised_point2.y);
  }

  cv::Mat points_4d;

  cv::triangulatePoints(T1, T2, points1, points2, points_4d);

  cout<<"printing points"<<endl;
  //for(cv::Point3d& point:points4D)
  for(int i=0; i < points_4d.cols; ++i)
  {
    cv::Mat point = points_4d.col(i);
    point /= point.at<double>(3, 0);
    points3D.emplace_back(point.at<double>(0, 0), point.at<double>(1, 0),
                         point.at<double>(2, 0));
    cout<<point.at<double>(0, 0)<<" "<<point.at<double>(1, 0)<<" "<<point.at<double>(2, 0)
        <<endl;
  }

}


int main()
{
  const cv::Mat img1 = cv::imread("1.png", 0);
  const cv::Mat img2 = cv::imread("2.png", 0);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  cv::Mat R, t;
  vector<cv::DMatch> matches;
  vector<cv::KeyPoint> kp1, kp2;

  find_R_and_t(img1, img2, K, R, t, matches, kp1, kp2);
  double t1 = t.at<double>(0, 0), t2 = t.at<double>(1, 0), t3 = t.at<double>(2, 0);

  cv::Mat t_x = (cv::Mat_<double>(3, 3) <<0.0, -t3, t2, t3, 0.0, -t1, -t2, t1, 0.0);
  cv::Mat t_xR = t_x * R;


  cout<<"epipolar constraint error"<<endl;
  for(const auto& match: matches)
  {
    cv::Point2d point1 = pixel2Cam(kp1[match.queryIdx].pt, K);
    cv::Point2d point2 = pixel2Cam(kp2[match.trainIdx].pt, K);

    cv::Mat y1 = (cv::Mat_<double>(3, 1)<<point1.x, point1.y, 1.0);
    cv::Mat y2 = (cv::Mat_<double>(3, 1)<<point2.x, point2.y, 1.0);
    cv::Mat error = y1.t() * t_xR * y2;
    cout<<error.t()<<endl;
  }

  vector<cv::Point3d> points3D;
  triangulate(kp1, kp2, matches, R, t, K, points3D);






  return 0;
}
