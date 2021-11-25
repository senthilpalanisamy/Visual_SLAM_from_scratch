#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <chrono>

using namespace std;

struct CURVE_FITTING_COST
{
  CURVE_FITTING_COST(double x, double y): _x(x), _y(y)
  {

  }

  template<typename T>
  bool operator() (const T *const abc, T *residual) const
  {
    // y - exp(a* x^2 + b * x + c)
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
    return true;
  }


  double _x, _y;
};


int main()
{
  vector<double> dataX, dataY;
  int N=100;
  cv::RNG rng;
  double sigma_w = 1.0;
  double inverse_sigma_w = 1.0 / sigma_w;
  vector<double> abcGT = {1.0, 2.0, 1.0};
  // vector<double*> abc = {2.0, -1.0, 5.0};
  double abc [] = {5.0, -10.0, 5.0};

  for(int i=0; i <N; ++i)
  {
    double x = static_cast<double>(i) / 100.0;
    dataX.emplace_back(x);
    double y = abcGT[0] * x * x + abcGT[1] * x  + abc[2] + rng(sigma_w * sigma_w);
    dataY.emplace_back(y);
  }



  
  ceres::Problem problem;
  for(int i=0; i < N; ++i)
  {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
          new CURVE_FITTING_COST(dataX[i], dataY[i])),
          nullptr,
          abc
        );
  }


  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
  cout<<"time: "<<time_used.count()<<" seconds"<<endl;
  cout<<summary.BriefReport()<<endl;
  cout<<"estimated a,b,c "<<abc[0]<<" "<<abc[1]<<" "<<abc[2]<<endl;
  cout<<"abc ground truth "<<abcGT[0]<<" "<<abcGT[1]<<" "<<abcGT[2]<<endl;


}
