#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <mutex>

#define PRINT_DEBUG 1

using namespace std;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;

double fx=718.856, fy=718.856, cx=607.1928, cy=185.2157;
double baseline = 0.573;


class JacobiAccumulator
{
  const Mat &referenceImage, &testImage;
  const VecVector2d &sampledPixels;
  VecVector2d projectedPixels;
  const VecVector3d &sampledPixel3DPositions;
  mutex hessianMutex;
  Sophus::SE3d &T_rt;
  Matrix6d H;
  Matrix26d J;
  Vector6d b;
  int goodCount;
  double intensityCost;

  public:
    JacobiAccumulator(const Mat &referenceImage_, const Mat &testImage_,
                     const VecVector2d &sampledPixels_, const VecVector3d &sampledPixel3DPositions_,
                     Sophus::SE3d& T_rt_):referenceImage(referenceImage_), testImage(testImage_),
                                         sampledPixels(sampledPixels_), sampledPixel3DPositions(sampledPixel3DPositions_),
                                         T_rt(T_rt_){
                                           goodCount = 0;
                                          // projectedPixels = vector<Eigen::Vector2d>(sampledPixels.size(), Eigen::Vector2d(-1, -1));
                                          projectedPixels = VecVector2d(sampledPixels.size(), Eigen::Vector2d(-1, -1));
                                          intensityCost = 0;
                                         }
    void accumulateJacobian(const Range &range);
    Matrix6d get_H()
    {
      return H;
    }

    Matrix26d get_J()
    {
      return J;
    }

    Vector6d get_b()
    {

      return b;
    }

    VecVector2d getProjections()
    {
      return projectedPixels;
    }

    double getCost()
    {
      if(goodCount != 0)
      {
        return intensityCost / goodCount;
      }
      else
      {
        return 0;
      }

    }

    void reset()
    {
      H = Matrix6d::Zero();
      J = Matrix26d::Zero();
      intensityCost = 0;
      goodCount = 0;
      //projectedPixels = VecVector2d(sampledPixels.size(), Eigen::Vector2d(-1, -1));
      //projectedPixels = vector<Eigen::Vector2d>(sampledPixels.size(), Eigen::Vector2d(-1, -1));
    }
};

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
  double interpolatedPixelValue = dxDash * dyDash * image.at<uchar>(yFloor, xFloor) +
                                  dxDash *dy * image.at<uchar>(yCeil, xFloor) +
                                  dx * dyDash * image.at<uchar>(yFloor, xCeil) +
                                  dx * dy * image.at<uchar>(yCeil, xCeil);
  return interpolatedPixelValue;
}

void JacobiAccumulator::accumulateJacobian(const Range& range)
{
  int halfPatchSize = 1;
  Matrix6d H_=Matrix6d::Zero();
  Vector6d b_ = Vector6d::Zero();
  int goodCount_=0;
  int rows = referenceImage.rows, cols = referenceImage.cols;
  //Matrix26d J_ = Matrix26d::Zero();
  double intensityCost_=0;


  for(size_t i=range.start; i < range.end; ++i)
  {
    const Eigen::Vector2d &referencePixel = sampledPixels[i];
    const Eigen::Vector3d &referencePixel3DPosition = sampledPixel3DPositions[i];


    Eigen::Vector3d P = T_rt * referencePixel3DPosition;
    double X = P[0], Y = P[1], Z=P[2];
    double ZInv = 1.0 / Z, Z2Inv = ZInv * ZInv;

    double u = fx * P[0] / P[2] + cx;
    double v = fy * P[1] / P[2] + cy;
    if(u < halfPatchSize || (u > cols - halfPatchSize ) || (v < halfPatchSize) ||
       v > (rows - halfPatchSize))
    {
      projectedPixels[i] = Eigen::Vector2d(-1, -1);
      continue;
    }
    goodCount_++;
    Matrix26d J_u_dT = Matrix26d::Zero();
    projectedPixels[i] = Eigen::Vector2d(u, v);

    J_u_dT(0, 0) = fx * ZInv;
    J_u_dT(1, 1) = fy * ZInv;
    J_u_dT(0, 2) = -fx * X * Z2Inv;
    J_u_dT(1, 2) = -fy * Y * Z2Inv;
    J_u_dT(0, 3) = -fx * X * Y * Z2Inv;
    J_u_dT(1, 3) = -fy - fy * Y * Y * Z2Inv;
    J_u_dT(0, 4) = fx + fx * X * X * Z2Inv;
    J_u_dT(1, 4) = fy * X * Y * Z2Inv;
    J_u_dT(0, 5) = -fx * Y * ZInv;
    J_u_dT(1, 5) = fy * X * ZInv;


    for(int x=-halfPatchSize; x <= halfPatchSize; ++x)
    {
      for(int y=-halfPatchSize; y <= halfPatchSize; ++y)
      {
        double intensityDifference = getPixelValue(referenceImage, referencePixel[1] + y, referencePixel[0] +x) - getPixelValue(testImage, v + y, u + x);
        intensityCost_ += intensityDifference * intensityDifference;

        Eigen::Vector2d J_i = Eigen::Vector2d::Zero();
        J_i(0, 0) = 0.5 * (getPixelValue(testImage, v + y, u + x+1) - getPixelValue(testImage, v + y, u + x-1));
        J_i(1, 0) = 0.5 * (getPixelValue(testImage, v + y+1, u + x) - getPixelValue(testImage, v+ y-1, u + x));

        Vector6d J_ = -(J_i.transpose() * J_u_dT).transpose();
        H_ += J_ * J_.transpose();
        b_ += -intensityDifference * J_;
      }
    }

  }

  {

    unique_lock<mutex> lock(hessianMutex);
    goodCount += goodCount_;
    if(goodCount_)
    {
    H += H_;
    b += b_;
    intensityCost += intensityCost_;
    }

  }





}


void directPoseEstimationSingleLayer(const Mat& referenceImage, const Mat& testImage,
                                     const VecVector2d &sampledPixels,
                                     const VecVector3d &sampledPixel3DPositions,
                                     Sophus::SE3d& T_rt)
  // Trt - transformation that transform point in reference image frame to test image frame
{
   JacobiAccumulator jacobi(referenceImage, testImage, sampledPixels, sampledPixel3DPositions, T_rt);

  int iterCount = 10;
  double lastCost=0, intensityCost=0;

  for(int iter=0; iter < iterCount; ++iter)
  {
    jacobi.reset();

  parallel_for_(Range(0, sampledPixels.size()), std::bind(&JacobiAccumulator::accumulateJacobian,
                      &jacobi, placeholders::_1));

   Matrix6d H = jacobi.get_H();
   Vector6d b = jacobi.get_b();
   double intensityCost = jacobi.getCost();
   Vector6d update = H.ldlt().solve(b);

   cout<<"current cost: "<<intensityCost<<endl;

   if(std::isnan(update[0]))
   {
     cout<<"Nan encountered in optimization"<<endl;
     break;
   }

   if((iter > 0) && (lastCost < intensityCost))
   {
     cout<<"cost increasing breaking"<<endl;
     break;
   }



   T_rt = Sophus::SE3d::exp(update) * T_rt;

   if(update.norm() < 1e-3)
   {
     cout<<"update too small breaking- Algorithm converged"<<endl;
     break;
   }

   lastCost = intensityCost;
  }


  Mat showImage;
  cvtColor(testImage, showImage, COLOR_GRAY2BGR);
  const VecVector2d &projections = jacobi.getProjections();
  for(int i=0; i < sampledPixels.size(); ++i)
  {
    const auto& currentPoint = projections[i];
    if(currentPoint[0] < 0 || currentPoint[1]< 0)
    {
      continue;
    }
    const auto& referencePoint = sampledPixels[i];
    // cout<<currentPoint[0]<<" "<<currentPoint[1]<<" "<<referencePoint[0]<<" "<<referencePoint[1]<<endl;
    circle(showImage, Point2f(currentPoint[0],  currentPoint[1]), 2, Scalar(255, 0, 0), 2);
    line(showImage, Point2f(referencePoint[0], referencePoint[1]), Point2f(currentPoint[0], currentPoint[1]),
                            Scalar(0, 255, 0));
  }
  imshow("current", showImage);
  waitKey(0);
}

void directPoseEstimationMultiLayer(const Mat& referenceImage, const Mat& testImage,
                                    const VecVector2d &sampledPixels,
                                    const VecVector3d &sampledPixel3DPositions,
                                    const Mat& disparity,
                                    Sophus::SE3d& T_rt)
{
  int pyramidLevels = 4;
  vector<double> pyramidScales = {0.125, 0.25, 0.5, 1.0};
  double scaleFactor = 0.5;

  vector<Mat> pyramidsRefImage, pyramidsTestImage;

  for(int i=0; i < pyramidLevels; ++i)
  {
    if(i == 0)
    {
      pyramidsRefImage.emplace_back(referenceImage);
      pyramidsTestImage.emplace_back(testImage);
    }
    else
    {
      Mat resizedRefImage, resizedTestImage;
      const Mat &refImage = pyramidsRefImage[pyramidsRefImage.size()-1];
      const Mat &testImage = pyramidsRefImage[pyramidsTestImage.size()-1];
      resize(refImage, resizedRefImage, Size(refImage.cols * scaleFactor, refImage.rows * scaleFactor));
      resize(testImage, resizedTestImage, Size(testImage.cols * scaleFactor, testImage.rows * scaleFactor));
      pyramidsRefImage.emplace_back(resizedRefImage);
      pyramidsTestImage.emplace_back(resizedTestImage);
    }
  }

  double fxG = fx , fyG = fy, cxG = cx, cyG = cy;

  for(int i=0; i < pyramidScales.size(); ++i)
  {
    double fx = fxG * pyramidScales[i], fy = pyramidScales[i] * fyG,
           cx = cxG * pyramidScales[i], cy = cyG * pyramidScales[i];
    VecVector2d rescaledPixels;
    VecVector3d newDepthPoints;

    for(auto pixel:sampledPixels)
    {
      rescaledPixels.emplace_back(pixel[0] * pyramidScales[i], pixel[1] * pyramidScales[i]);

      double thisDisparity = disparity.at<uchar>(pixel[1], pixel[0]);

      double depth = (fx * baseline)  / thisDisparity;
      double X = depth * (rescaledPixels[rescaledPixels.size()-1][0] - cx) / fx;
      double Y = depth * (rescaledPixels[rescaledPixels.size()-1][1] - cy) / fy;


      // sampledPixel3DPositions.emplace_back(X, Y, depth);
      newDepthPoints.emplace_back(X, Y, depth);
    }


    // directPoseEstimationSingleLayer(pyramidsRefImage[pyramidLevels-1-i], pyramidsTestImage[pyramidLevels-1-i], rescaledPixels,
    //                                 sampledPixel3DPositions, T_rt);
    cout<<fx<<" "<<fy<<" "<<cx<<" "<<cy<<endl;

    directPoseEstimationSingleLayer(pyramidsRefImage[pyramidLevels-1-i], pyramidsTestImage[pyramidLevels-1-i], rescaledPixels,
                                    newDepthPoints, T_rt);
  }
}



int main()
{
  const Mat referenceImage = imread("./left.png", 0);
  const Mat disparity = imread("./disparity.png", 0);
  boost::format imageSequenceFormat("./%06d.png");
  //vector<Mat> testImages;
  int numOfImages = 5;
  int numOfPoints = 2000;
  VecVector3d sampledPixel3DPositions;
  VecVector2d sampledPixels;
  int height = referenceImage.rows;
  int width = referenceImage.cols;
  int border = 20;
  RNG rng;
  Sophus::SE3d T_rt;

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
    sampledPixel3DPositions.emplace_back(X, Y, depth);
  }

  for(int i=1; i <= numOfImages; ++i)
  {
    string fileName = (imageSequenceFormat % i).str();
    Mat testImage = imread((imageSequenceFormat % i).str(), 0);
    directPoseEstimationSingleLayer(referenceImage, testImage, sampledPixels, sampledPixel3DPositions, T_rt);
    //directPoseEstimationMultiLayer(referenceImage, testImage, sampledPixels, sampledPixel3DPositions, disparity, T_rt);

  }




  return 0;
}
