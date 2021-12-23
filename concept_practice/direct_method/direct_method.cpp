#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix>double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;

double fx=718.856, fy=718.856, cx=607.1928, cy=185.2157;
double baseline = 0.573;


class JacobiAccumulator
{
  const Mat &referenceImage, &tesImage;
  const VecVector2d &sampledPixels;
  const VecVector3d &samopledPixel3DPositions;
  Sophus::SE3d T_rt;
  Matrix6d H;
  Matrix26d J;
  Vector6d b;

  public:
    JacobiAccumualtor(const Mat &referenceImage_, const Mat &testImage_,
                     const VecVector2d &sampledPixels_, const VecVector3d &sampledPixel3DPositions_,
                     Sophus::SE3d T_rt_):referenceImage(referenceImage_), testImage(testImage_),
                                         sampledPixels(sampledPixels_), sampledPixel3DPositions(sampledPixel3DPositions),
                                         T_rt(T_rt_){}
    void accumulateJacobian(Range &range);
    Matrix6d get_H()
    {
      return H;
    }

    Matrix26d get_J()
    {
      return J
    }

    Vector6d get_b()
    {
      return b;
    }
}

double getPixelValue(const Mat &image, double y, double x)
{
  int height = image.rows;
  int width = image.cols;

  if(y > height-1)
  {
    y = height - 1;
  }

  if(y < 0)
  {
    y = 0;
  }

  if(x > width-1)
  {
    x = width-1;
  }

  if(x < 0)
  {
    x = 0;
  }

  int xFloor = int(floor(x));
  int xCeil = min(int(ceil(x)), width-1);
  int yFloor = int(floor(y));
  int yCeil = min(int(ceil(y)), height-1);
  double dx = x - xFloor;
  double dxDash = 1 - dx;
  double dy = y - yFloor;
  double dyDash = 1 - dy;
  double interpolatedPixelValue = dxDash * dyDash * image[yFloor, xFloor] +
                                  dxDash *dy * image[yCeil, xFloor] +
                                  dx * dyDash * image[yFloor, xCeil] +
                                  dx * dy * image[yCeil, xCeil];
  return interpolatedPixelValue;
}

void JacobiAccumulator::accumulateJacobian(Range& range)
{
  int halfPatchSize = 1;
  Matrix6d H_=Matrix6d::zero();
  Vector6d b = Vector6d::zero();
  //Matrix26d J_ = Matrix26d::zero();


  for(size_t i=range.start; i < range.end; ++i)
  {
    const Eigen::Vector2d &referencePixel = sampledPixels[i];
    const Eigen::Vector3d &referencePixel3DPosition = sampledPixel3DPositions[i];

    Eigen::Vector3d P = T_rt * referencePixel3DPosition;
    double X = P[0], Y = P[1], Z=P[2];
    double ZInv = 1 / Z, Z2Inv = ZInv * ZInv;

    double u = fx * P[0] / P[2] + cx;
    double v = fy * P[1] / P[2] + cy;
    double error=0;
    Matrix26d J_u_dT = Matrix26d::zero();

    J_u_dT(0, 0) = fx * ZInv;
    J_u_dT(1, 1) = fy * ZInv;
    J_u_dT(0, 2) = -fx * X * Z2Inv;
    J_u_dt(1, 2) = -fy * Y * Z2Inv;
    J_u_dt(0, 3) = -fx * X * Y * Z2Inv;
    J_u_dt(1, 3) = -fy - fy * Y * Y * Z2Inv;
    J_u_dt(0, 4) = fx + fx * X * X * Z2Inv;
    J_u_dt(1, 4) = fy * X * Y * Z2Inv;
    J_u_dt(0, 5) = -fx * Y * ZInv;
    J_u_dt(1, 5) = fy * X * ZInv;


    for(int x=-halfPatchSize; x <= halfPatchSize; ++x)
    {
      for(int y=-halfPatchSize; y <= halfPatchSize; ++y)
      {
        intensityDifference = getPixelValue(referenceImage, y, x) - getPixelValue(testImage, y, x);
        error += intensityDifference * intensityDifference;

        Eigen::Vector2d J_i = Eigen::Vector2d::zero();
        J_i(0, 0) = 0.5 * (getPixelValue(testImage, y, x-1) - getPixelValue(testImage, y, x+1));
        J_i(1, 0) = 0.5 * (getPixelValue(testImage, y-1, x) - getPixelValue(testImage, y+11, x));

        Vector6d J_ = (J_i.transpose() * J_u_dt).tranpose();
        H_ += J_ * J_.transpose();
        b += error * J;
      }
    }

  }




}


void directPoseEstimationSingleLayer(const Mat& referenceImage, Mat& testImage,
                                     const VecVector2d &sampledPixels,
                                     const VecVector3d &sampledPixel3DPositions,
                                     Sophus::SE3d T_rt)
  // Trt - transformation that transform point in reference image frame to test image frame
{




}


int main()
{
  const Mat leftImage = imread("./left.png", 0);
  const Mat disparity = imread("./disparity.png", 0);
  boost::format imageSequenceFormat("./%o6d.png");
  //vector<Mat> testImages;
  int numOfImages = 6;
  int numOfPoints = 2000;
  VecVector3d sampledPixel3DPositions;
  VecVector2d sampledPixels;
  int height = leftImage.rows;
  int width = leftImage.cols;
  int border = 20;
  RNG rng;

  for(int i=0; i < numOfPoints; ++i)
  {
    int x = rng.uniform(border, width - border);
    int y = rng.uniform(border, height - border);
    double thisDisparity = disparity.at<uchar>(y, x);
    if(thisDisparity < 0)
    {
      continue;
    }

    double depth = (fx * baseline)  / thisDisparity;
    double X = depth * (x - cx) / fx;
    double Y = depth * (y - cy) / fy;

    sampledPixels.emplace_back(x, y);
    sampledPixel3DPositions.emplace_back(X, y, depth);
  }

  for(int i=0; i < numOfImages; ++i)
  {
    Mat testImage = imread((imageSequenceFormat % i).str(), 0);
    directPoseEstimationSingleLayer()
  }




  return 0;
}
