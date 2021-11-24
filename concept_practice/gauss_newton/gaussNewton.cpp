#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace Eigen;


int main()
{
  double aGT=1.0, bGT=2.0, cGT=1.0;
  double ar=2.0, br = -1.0, cr = 5.0;

  vector<double> xData, yData;
  int N=100;
  cv::RNG rng;
  double w_sigma = 1.0;
  double w_sigma_inverse = 1.0 / w_sigma;
  int maxIterations = 100;

  for(int i=0; i < N; ++i)
  {
    double x = i / 100.0;
    xData.emplace_back(x);
    yData.emplace_back(exp(aGT * x * x + bGT * x + cGT)+ rng(w_sigma * w_sigma));
  }


  double cost=0, lastCost=0;
 chrono::steady_clock::time_point t1 = chrono::steady_clock::now();


 for(int iter=0; iter < maxIterations; ++iter)
 {
   Vector3d b = Vector3d::Zero();
   Matrix3d H = Matrix3d::Zero();
   cost =0;

   for(int i=0; i < N; ++i)
   {

     Vector3d J = Vector3d::Zero();
     double functionValue = exp(ar * xData[i] * xData[i] + br * xData[i] + cr);
     double error = yData[i] - functionValue;
     J[0] -= xData[i] * xData[i] * functionValue;
     J[1] -= xData[i] * functionValue;
     J[2] -= functionValue;
     H += w_sigma_inverse * w_sigma_inverse * J * J.transpose();
     b += -w_sigma_inverse * w_sigma_inverse * J * error;
     cost += error * error;
   }

   Vector3d deltaX = H.ldlt().solve(b);
   if(isnan(deltaX[0]))
   {
     cout<<"nan encountered in step size calculation"<<endl;
   }

   if(iter > 0 && cost > lastCost)
   {
     cout<<"cost function not decreasing "<<cost<<"  "<<lastCost<<endl;
     break;
   }
   ar = ar + deltaX[0];
   br = br + deltaX[1];
   cr = cr + deltaX[2];

   //cout<<"error at iteration "<<iter<<" :"<<cost<<"last error: "<<lastCost<<"\n";
   lastCost = cost;
 }

 chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
 chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
 cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
 cout<<"final values: a: "<<ar<<" b:"<<br<<" c:"<<cr<<endl;
 cout<<"original values: a: "<<aGT<<" b:"<<bGT<<" c:"<<cGT<<endl;


  return 0;
}
