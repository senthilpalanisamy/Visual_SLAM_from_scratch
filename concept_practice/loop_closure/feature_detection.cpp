#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include<vector>
#include<string>

#include "DBoW3/DBoW3.h"

using namespace std;
using namespace cv;

int main(int arg, char** argv)
{
  string data_directory_path = argv[1];
  Ptr<Feature2D> detector = ORB::create();
  vector<Mat> descriptors;

  for(int i=0; i < 10; ++i)
  {
    string path = data_directory_path+ "/" + to_string(i+1) +".png";
    Mat image = imread(path);

    vector<KeyPoint> keyPoints;
    Mat descriptor;
    detector->detectAndCompute(image, Mat(), keyPoints, descriptor);
    descriptors.push_back(descriptor);
  }

  DBoW3::Vocabulary vocab;
  vocab.create(descriptors);
  vocab.save("vocabulary.yml.gz");
  cout<<"finsihed"<<endl;

}
