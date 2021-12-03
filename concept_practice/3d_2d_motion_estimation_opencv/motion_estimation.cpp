#include<opencv2/opencv.hpp>
#include "feature_matching.hpp"

using namespace std;



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
  const cv::Mat depth1 = cv::imread("1_depth.png", cv::IMREAD_UNCHANGED);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<cv::DMatch> matches;
  vector<cv::KeyPoint> kp1, kp2;

  match_image_features(img1, img2, matches, kp1, kp2);
  vector<cv::Point3d> points_3d;
  vector<cv::Point2d> points_2d;

  for(const auto& match: matches)
  {
    ushort depth = depth1.at<unsigned short>(int(kp1[match.queryIdx].pt.y), int(kp1[match.queryIdx].pt.x));
    if(depth == 0)
      continue;
    double scaledDepth = depth / 5000.0;

    cv::Point2d normalisedPoint = pixel2Cam(kp1[match.queryIdx].pt, K);
   points_3d.push_back(cv::Point3d(normalisedPoint.x * scaledDepth,
                                   normalisedPoint.y * scaledDepth,
                                   scaledDepth));
   points_2d.emplace_back(kp2[match.trainIdx].pt.x, kp2[match.trainIdx].pt.y);
  }

  cv::Mat r,t;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  cv::solvePnP(points_3d, points_2d, K, cv::Mat(), r, t, false);
  cv::Mat R;
  cv::Rodrigues(r, R);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout<<" time for solve pnp: "<<time_used.count()<<" seconds"<<endl;
  cout<<"R = "<<endl<<R<<endl;
  cout<<"t="<<endl<<t<<endl;

  return 0;
}
