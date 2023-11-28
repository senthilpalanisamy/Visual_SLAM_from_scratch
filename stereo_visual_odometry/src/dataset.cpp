#include <fstream>
#include <stdexcept>
#include <iostream>

#include <stereo_slam/dataset.hpp> 
using std::string;
using std::cout;


KittiDatasetAdapter::KittiDatasetAdapter(const string& dataPath_):dataPath(dataPath_)
{
  string calibPath = dataPath + "/" + calibFile;
  cout<<"path:"<<calibPath<<std::endl;
  imagePathFormat = boost::format("%s/image_%d/%06d.png");
  std::ifstream fin(calibPath);
  imageIndex=0;
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
    Mat33 K;
    K << projectionMatrix[0], projectionMatrix[1], projectionMatrix[2], projectionMatrix[4], projectionMatrix[5], projectionMatrix[6], projectionMatrix[8], projectionMatrix[9], projectionMatrix[10];

  Vec3 t;
  t << projectionMatrix[3], projectionMatrix[7], projectionMatrix[11];
  t = K.inverse() * t;
  K = K * downsamplingFactor;
  Camera::Ptr newCamera(new Camera(projectionMatrix[0], projectionMatrix[5], projectionMatrix[3], projectionMatrix[7], SE3(SO3(), t)));
  cameras.push_back(newCamera);
  }
}

Frame::Ptr KittiDatasetAdapter::nextFrame()
{

  auto leftImagePath = (imagePathFormat % dataPath % 0 % imageIndex).str();
  auto rightImagePath = (imagePathFormat % dataPath % 1 % imageIndex).str();
  Frame::Ptr newFrame = Frame::createFrame();
  cv::Mat leftImage, rightImage;
  leftImage = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
  rightImage = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);
  newFrame->rightImage = rightImage;
  cv::Mat image_left_resized, image_right_resized;
  cv::resize(leftImage, image_left_resized, cv::Size(), 0.5, 0.5,
                 cv::INTER_NEAREST);
  cv::resize(rightImage, image_right_resized, cv::Size(), 0.5, 0.5,
                 cv::INTER_NEAREST);


  newFrame->leftImage = image_left_resized;
  newFrame->rightImage = image_right_resized;

  cout<<"type: "<<newFrame->leftImage.type()<<std::endl;
  cout<<"type: "<<newFrame->rightImage.type()<<std::endl;
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
