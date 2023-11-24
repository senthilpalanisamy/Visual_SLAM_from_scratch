#include<stereo_slam/visual_odometry.hpp>
#include<stereo_slam/frontend.hpp>

#include<iostream>
using std::cout;
using std::endl;


void SVO::run()
{
  while(true)
  {
    auto nextFrame = dataset.nextFrame();
    if(nextFrame == nullptr)
    {
      cout<<"exiting SVO"<<std::endl;
      return;
    }
    frontend.addFrame(nextFrame);

  }
}
