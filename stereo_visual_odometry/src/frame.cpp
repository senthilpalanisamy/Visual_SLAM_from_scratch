#include <stereo_slam/frame.hpp>

Frame::Ptr Frame::createFrame()
{
  return shared_ptr<Frame>(new Frame());
}
