#include<iostream>
#include<vector>
#include<string>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "DBoW3/DBoW3.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
  DBoW3::Vocabulary vocab("./vocabulary.yml.gz");
  if(vocab.empty())
  {
    cerr<<"vocab is empty"<<endl;
    return 1;
  }
  string data_directory = argv[1];
  
  vector<Mat> descriptors(10);
  Ptr<Feature2D> detector = ORB::create();

  for(int i=1; i <=10; ++i)
  {
    string imagePath = data_directory + "/" + to_string(i) + ".png";
    Mat image = imread(imagePath);
    vector<KeyPoint> keyPoints;
    Mat descriptor;
    // detector->detectAndCompute(image, Mat(), keyPoints, descriptors[i-1]);
    detector->detectAndCompute(image, Mat(), keyPoints, descriptors[i-1]);
    // descriptors.emplace_back(descriptor);
  }


  for(int i=0; i < 10; ++i)
  {
    DBoW3::BowVector v1;
    vocab.transform(descriptors[i], v1);
    cout<<"##############################"<<endl;
    for(int j=0; j < 10; ++j)
    {
      DBoW3::BowVector v2;
      vocab.transform(descriptors[j], v2);
      double score = vocab.score(v1, v2);
      cout<<"distance between image "<<i+1<<" "<<j+1<<":"<<score<<endl;
    }
  }

  DBoW3::Database db(vocab, false, 0);
  for(int i=0; i < descriptors.size(); ++i)
  {
    db.add(descriptors[i]);
  }

  cout<<"top 4 matches for each image from database";

  for(int i=0; i < 10; ++i)
  {
    DBoW3::QueryResults ret;
    db.query(descriptors[i], ret, 4);
    cout<<"results for image "<<i<<" "<<endl;
    cout<<ret<<endl<<endl;
  }




}


