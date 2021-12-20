#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>


using namespace std;
using namespace cv;

class OpticalFlowTracker
{

  public:
  OpticalFlowTracker(const Mat& image1_, const Mat& image2_,
                    const vector<KeyPoint>& kp1_, vector<KeyPoint>& kp2_,
                    vector<bool> &success_, bool inverse_=true,
                    bool has_initial_=false):img1(image1_),
                    img2(image2_), kp1(kp1_), kp2(kp2_), success(success_),
                    inverse(inverse_), has_initial(has_initial_){}
  void calculateOpticalFlow(const Range& range);

  private:
  const Mat &img1;
  const Mat &img2;
  const vector<KeyPoint> &kp1;
  vector<KeyPoint> &kp2;
  vector<bool> &success;
  bool inverse, has_initial;
};


// inline float getPixelValue(const cv::Mat &img, float y, float x) {                                                                                                                                       
//       // boundary check                                                                                                                                                                                    
//       if (x < 0) x = 0;                                                                                                                                                                                    
//       if (y < 0) y = 0;                                                                                                                                                                                    
//       if (x >= img.cols - 1) x = img.cols - 2;                                                                                                                                                             
//       if (y >= img.rows - 1) y = img.rows - 2;                                                                                                                                                             
//                                                                                                                                                                                                            
//       float xx = x - floor(x);                                                                                                                                                                             
//       float yy = y - floor(y);                                                                                                                                                                             
//       int x_a1 = std::min(img.cols - 1, int(x) + 1);                                                                                                                                                       
//       int y_a1 = std::min(img.rows - 1, int(y) + 1);                                                                                                                                                       
//                                                                                                                                                                                                            
//       return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)                                                                                                                                                     
//       + xx * (1 - yy) * img.at<uchar>(y, x_a1)                                                                                                                                                             
//       + (1 - xx) * yy * img.at<uchar>(y_a1, x)                                                                                                                                                             
//       + xx * yy * img.at<uchar>(y_a1, x_a1);                                                                                                                                                               
//   }            

int getPixelValue2(const Mat& img, float y, float x)
{
  if(x < 0)
  {
    x = 0;
  }

  if(x >= img.cols-1)
  {
    x = img.cols - 2;
  }

  if(y < 0)
  {
    y = 0;
  }

  if(y >= img.rows-1)
  {
    y = img.rows - 2;
  }

  float xx = x - floor(x);
  float yy = y - floor(y);
  int xNext = std::min(img.cols-1, int(floor(x))+1);
  int yNext = std::min(img.rows-1, int(floor(y))+1);
  int xPrev = int(floor(x));
  int yPrev = int(floor(y));

  return (1 - xx) * (1 - yy) * img.at<uchar>(yPrev, xPrev) +
         xx * (1- yy) * img.at<uchar>(yPrev, xNext) +
         yy * (1 - xx) * img.at<uchar>(yNext, xPrev) +
         xx * yy * img.at<uchar>(yNext, xNext);

}


void OpticalFlowTracker::calculateOpticalFlow(const Range& range)
{
  int half_patch_size = 4;
  int iterations = 10;

  for(size_t i=range.start; i < range.end; i++)
  {
    auto kp = kp1[i];
    double dx=0, dy=0;

    if(has_initial)
    {
      dx = kp2[i].pt.x - kp.pt.x;
      dy = kp2[i].pt.y - kp.pt.y;
    }


    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b = Eigen::Vector2d::Zero();
    Eigen::Vector2d J;
    double cost=0, lastCost=0;
    bool succ = true;

    for(int iter=0; iter < iterations; ++iter)
    {
      if(inverse == false)
      {
        H = Eigen::Matrix2d::Zero();
        b = Eigen::Vector2d::Zero();
      }
      else
      {
        b = Eigen::Vector2d::Zero();
      }

    cost=0;


      for(int x=-half_patch_size; x < half_patch_size; ++x)
      {
        for(int y=-half_patch_size; y < half_patch_size; ++y)
        {
          double error = getPixelValue(img1, kp1[i].pt.y + y, kp1[i].pt.x + x) -
                         getPixelValue(img2, kp1[i].pt.y + y + dy, kp1[i].pt.x + x + dx);
          if(inverse == false)
          {
            J = - 0.5 * Eigen::Vector2d(getPixelValue(img2, kp1[i].pt.y + dy + y, kp1[i].pt.x + dx +x+1) -
                                       getPixelValue(img2, kp1[i].pt.y + dy + y, kp1[i].pt.x + dx + x -1),
                                       getPixelValue(img2, kp1[i].pt.y + dy + y +1, kp1[i].pt.x + dx +  x ) -
                                       getPixelValue(img2, kp1[i].pt.y + dy + y -1, kp1[i].pt.x + dx +x));

          }
          else if(iter==0)
          {
            J = -0.5 * Eigen::Vector2d(getPixelValue(img1, kp1[i].pt.y+y, kp1[i].pt.x + x +1) -
                                       getPixelValue(img1, kp1[i].pt.y+y, kp1[i].pt.x + x -1),
                                       getPixelValue(img1, kp1[i].pt.y+y+1, kp1[i].pt.x+x) -
                                      getPixelValue(img1, kp1[i].pt.y+y-1, kp1[i].pt.x+x));
          }

          b += - error * J;
          cost += error * error;
          if(inverse == false || iter== 0)
          {
            H += J * J.transpose();
          }
       }

      }

          Eigen::Vector2d update = H.ldlt().solve(b);

          if(std::isnan(update[0]))
          {
            succ = false;
            cout<<"nan during update"<<endl;
          }

          if((cost > lastCost) && (iter != 0))
          {
            cout<<"cost increasing breaking from loop"<<endl;
            cout<<"last cost: "<<lastCost<<" this cost: "<<cost<<endl;
            break;
          }

          dx += update[0];
          dy += update[1];
          lastCost = cost;
          succ  = true;
          cout<<"curernt cost: "<<cost<<endl;
          if(update.norm() < 1e-2)
          {
            cout<<"converged"<<endl;
            break;
          }


        }

          kp2[i].pt.y = kp1[i].pt.y + dy;
          kp2[i].pt.x = kp1[i].pt.x + dx;

          success[i] = succ;
      }


    }




void calculateSingleLayerOpticalFlow(const Mat& image1, const Mat& image2,
                                     const vector<KeyPoint>& kp1,
                                     vector<KeyPoint>& kp2,
                                     vector<bool>& success,
                                     bool inverse=false, bool has_initial=false)
{
  kp2.resize(kp1.size());
  success.resize(kp1.size());

  OpticalFlowTracker tracker(image1, image2, kp1, kp2, success, inverse, has_initial);
  parallel_for_(Range(0, kp1.size()), std::bind(&OpticalFlowTracker::calculateOpticalFlow,
                      &tracker, placeholders::_1));
}


void calculateMultiLayerOpticalFlow(const Mat& image1, const Mat& image2,
                                   const vector<KeyPoint>& kp1,
                                   vector<KeyPoint>& kp2,
                                   vector<bool>& success)
{
  int pyramids = 4;
  double pyramid_scale = 0.5;
  double scales[] = {1.0, 0.5, 0.25, 0.125};
  vector<Mat> pyr1, pyr2;

  for(int i=0; i < pyramids; ++i)
  {
    if(i == 0)
    {
      pyr1.emplace_back(image1);
      pyr2.emplace_back(image2);
    }
    else
    {
      Mat img1Resize, img2Resize;
      resize(pyr1[i-1], img1Resize,
             Size(pyr1[i-1].cols * pyramid_scale, pyr1[i-1].rows * pyramid_scale));
      resize(pyr2[i-1], img2Resize,
             Size(pyr2[i-1].cols * pyramid_scale, pyr2[i-1].rows * pyramid_scale));
      pyr1.emplace_back(img1Resize);
      pyr2.emplace_back(img2Resize);
    }
 }

  vector<KeyPoint> kp_img1, kp_img2;
  for(auto kp: kp1)
  {
   kp.pt *= scales[pyramids -1];
   kp_img1.emplace_back(kp);
   kp_img2.emplace_back(kp);
  }

  for(int i=pyramids - 1; i >=0; --i)
  {
    success.clear();
    calculateSingleLayerOpticalFlow(pyr1[i], pyr2[i], kp_img1, kp_img2, success, false, true);
    if(i > 0)
    {
      for(auto& kp: kp_img1)
      {
        kp.pt /= pyramid_scale;
      }

      for(auto& kp: kp_img2)
      {
        kp.pt /= pyramid_scale;
      }
    }

  }

  kp2 = kp_img2;
}



int main()
{
  string image1Path = "./LK1.png";
  string image2Path = "./LK2.png";
  Mat image1 = imread(image1Path);
  Mat image2 = imread(image2Path);

  vector<KeyPoint> kp1;
  Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
  detector->detect(image1, kp1);

  vector<KeyPoint> kp2;
  vector<bool> success;
  //calculateSingleLayerOpticalFlow(image1, image2, kp1, kp2, success);
  calculateSingleLayerOpticalFlow(image1, image2, kp1, kp2, success);
  //calculateMultiLayerOpticalFlow(image1, image2, kp1, kp2, success);


  vector<cv::DMatch> matches;
  for(int i=0; i < 10; ++i)
  {
    if(success[i])
    {
      cv::DMatch match(i, i, 0);
      matches.emplace_back(match);
    }

  }
  cout<<"matching size: "<<matches.size()<<endl;
  cout<<"key point size: "<<kp1.size()<<endl;


  Mat outputImage;

  cv::drawMatches(image1, kp1, image2, kp2, matches, outputImage);
  cv::imshow("output image", outputImage);
  cv::waitKey(0);

}
