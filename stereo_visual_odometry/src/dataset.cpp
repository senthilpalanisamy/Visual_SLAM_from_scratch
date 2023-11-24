#include <fstream>
#include <stdexcept>
#include <iostream>

#include <stereo_slam/dataset.hpp> 
using std::string;
using std::cout;

Camera::Camera(double fx_, double fy_, double cx_, double cy_):fx(fx_), fy(fy_), cx(cx_), cy(cy_)
{

};

KittiDatasetAdapter::KittiDatasetAdapter(const string& dataPath_):dataPath(dataPath_)
{
  string calibPath = dataPath + "/" + calibFile;
  cout<<"path:"<<calibPath<<std::endl;
  std::ifstream fin(calibPath);
  if(!fin)
  {
    throw std::invalid_argument("file path doesn't exist");
  }
  char cameraName[3];
  double projectionMatrix[12];
  for(int i=0; i < 4; ++i)
  {
    for(int j=0; j < 3; ++j)
    {
      fin >> cameraName[j];
    }

    for(int j=0; j < 12; ++j)
    {
      fin >>projectionMatrix[j];
    }
  }
  Camera::Ptr newCamera(new Camera(projectionMatrix[0], projectionMatrix[5], projectionMatrix[3], projectionMatrix[7]));
  cameras.push_back(newCamera);
  imageIndex=0;
  imagePathFormat = boost::format("%s/image_%d/%06d.png");
}

Frame::Ptr KittiDatasetAdapter::nextFrame()
{

  auto leftImagePath = (imagePathFormat % dataPath % 0 % imageIndex).str();
  auto rightImagePath = (imagePathFormat % dataPath % 1 % imageIndex).str();
  Frame::Ptr newFrame = Frame::createFrame();
  newFrame->leftImage = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
  newFrame->rightImage = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);
  if((newFrame->leftImage.data == nullptr) || (newFrame->rightImage.data == nullptr))
  {
    return nullptr;
  }
  cv::imshow("leftImage", newFrame->leftImage);
  cv::imshow("rightImage", newFrame->rightImage);
  cv::waitKey(0);

  ++imageIndex;
  return newFrame;

}
