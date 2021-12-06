#ifndef FEATURE_MATCHING_INCLUDE_DIR
#define FEATURE_MATCHING_INCLUDE_DIR

#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <nmmintrin.h>

using namespace std;

void match_image_features(const cv::Mat& img1, const cv::Mat& img2,
                          vector<cv::DMatch>& matches,
                          vector<cv::KeyPoint>& kp1,
                          vector<cv::KeyPoint>& kp2, bool debug=false);


#endif
