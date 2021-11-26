#include<Eigen/Core>
#include<iostream>
#include<g2o/core/g2o_core_api.h>
#include<g2o/core/base_vertex.h>
#include<g2o/core/block_solver.h>
#include<g2o/core/base_unary_edge.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/core/optimization_algorithm_gauss_newton.h>
#include<g2o/core/optimization_algorithm_dogleg.h>
#include<g2o/solvers/dense/linear_solver_dense.h>
#include<opencv2/core/core.hpp>
#include<cmath>
#include<chrono>




using namespace std;


class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d> {

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      virtual void setToOriginImpl() override {
        _estimate <<0, 0, 0;
      }

    virtual void oplusImpl(const  double *update) override{
      _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(istream &in){}
    virtual bool write(ostream &out) const {}
};


class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>{

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x):BaseUnaryEdge(), _x(x){}

    virtual void computeError() override {
      const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
      const Eigen::Vector3d abc= v->estimate();
      _error(0,0) = _measurement - std::exp(abc(0,0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    virtual void linearizeOplus() override{
      const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
      const Eigen::Vector3d abc = v->estimate();
      double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
      _jacobianOplusXi[0] = - _x * _x * y;
      _jacobianOplusXi[1] = - _x * y;
      _jacobianOplusXi[2] = -y;
    }

    virtual bool read(istream& in){}
    virtual bool write(ostream& out) const {}


    double _x;
};


int main()
{
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);


  vector<double> dataX, dataY;
  int N=100;
  cv::RNG rng;
  double sigma_w = 1.0;
  double inverse_sigma_w = 1.0 / sigma_w;
  vector<double> abcGT = {1.0, 2.0, 1.0};
  // vector<double*> abc = {2.0, -1.0, 5.0};
  double abc [] = {2.0, -1.0, 5.0};

  for(int i=0; i <N; ++i)
  {
    double x = static_cast<double>(i) / 100.0;
    dataX.emplace_back(x);
    double y = abcGT[0] * x * x + abcGT[1] * x  + abc[2] + rng(sigma_w * sigma_w);
    dataY.emplace_back(y);
  }

  CurveFittingVertex *v = new CurveFittingVertex();
  v->setEstimate(Eigen::Vector3d(abc[0], abc[1], abc[2]));
  v->setId(0);
  optimizer.addVertex(v);

  for(int i=0; i < N; ++i)
  {
    CurveFittingEdge* edge = new CurveFittingEdge(dataX[i]);
    edge->setId(i);
    edge->setVertex(0, v);
    edge->setMeasurement(dataY[i]);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0 / (sigma_w * sigma_w));
    optimizer.addEdge(edge);
  }

  cout<<"optimisation started\n";
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
  cout<<"solving time: "<<time_used.count()<<"seconds "<<endl;

  Eigen::Vector3d estimatedabc = v->estimate();
  cout<<"estimated values: "<<estimatedabc.transpose()<<endl;
  cout<<"original values: "<<abcGT[0]<<" "<<abcGT[1]<<" "<<abcGT[2];


  return 0;
}
