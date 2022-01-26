#ifndef SnavelyReprojectionError_INCLUDE_DIR
#define SnavelyReprojectionError_INCLUDE_DIR

#include <ceres/ceres.h>

#include "snavelyReprojectionError.h"
#include "rotation.h"

class SnavelyReprojectionError
{
  public:

  SnavelyReprojectionError(double x_, double y_):x(x_), y(y_)
  {

  }

  template <typename T>
  bool operator ()(const T *const camera,
                   const T *const point,
                   T *residuals) const
  {
    T predictions[2];
    camProjectionWithDistortion(camera, point, predictions);
    residuals[0] = predictions[0] - T(x);
    residuals[1] = predictions[1] - T(y);

    return true;
  }

  template<typename T>
  static inline bool camProjectionWithDistortion(const T *const camera, const T *const point,
                                   T *predictions)
  {
    T p[3];
    AngleAxisRotatePoint(camera, point, p);

    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T xp = -p[0] / p[2];
    T yp = -p[1] / p[2];

    const T &k1 = camera[7];
    const T &k2 = camera[8];

    const T r2 = xp * xp + yp * yp;
    T xd = (T(1.0) + k1 * r2 + k2 * r2 * r2) * xp;
    T yd = (T(1.0) + k1 * r2 + k2 * r2 * r2) * yp;

    const T &focal = camera[6];
    predictions[0] = focal * xd;
    predictions[1] = focal * yd;
    return true;
  }

  static ceres::CostFunction *Create(const double x, const double y)
 {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
          new SnavelyReprojectionError(x, y)
          ));
  }

  private:
  double x, y;

};


#endif
