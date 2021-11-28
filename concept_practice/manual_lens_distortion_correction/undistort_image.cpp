#include <opencv2/opencv.hpp>


using namespace std;


int main()
{

  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

  cv::Mat image = cv::imread("distorted.png", 0);
  cv::Mat image_undistort = cv::Mat(image.rows, image.cols, CV_8UC1);

  for(int v=0; v < image_undistort.rows; ++v)
  {
    for(int u=0; u < image_undistort.cols; ++u)
    {

      double x = (u - cx) / fx;
      double y = (v - cy) / fy;

      double r = sqrt(x*x + y*y);

      double x_d = (1 + k1 * r *r + k2 * r * r * r * r) * x +
               2 * p1 * x * y + p2 * (r * r + 2 * x * x);
      double y_d = (1 + k1 * r * r + k2 * r * r * r * r) * y +
               p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;

      x_d = fx * x_d + cx;
      y_d = fy * y_d + cy;

      if(x_d >=0 && x_d < image_undistort.cols && y_d>=0 && y_d < image_undistort.rows)
      {
        image_undistort.at<uchar>(v, u) = image.at<uchar>((int) y_d, (int) x_d);
      }
    }
  }


  cv::imshow("distorted image", image);
  cv::imshow("undistorted image", image_undistort);
  cv::waitKey(0);







  return 0;
}
