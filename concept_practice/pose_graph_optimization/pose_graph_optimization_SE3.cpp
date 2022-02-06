#include<iostream>
#include<fstream>
#include<string>
#include<Eigen/Core>


#include<g2o/core/base_vertex.h>
#include<g2o/core/base_binary_edge.h>
#include<g2o/core/block_solver.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/solvers/eigen/linear_solver_eigen.h>

#include<sophus/se3.hpp>

using namespace Eigen;
using namespace std;
using Sophus::SE3d;
using Sophus::SO3d;

typedef Matrix<double, 6, 6> Matrix6d;

Matrix6d computeJInv(const SE3d T)
{
  Matrix6d JInv;
  JInv.block(0, 0, 3, 3) = SO3d::hat(T.so3().log());
  JInv.block(0, 3, 3, 3) = SO3d::hat(T.translation());
  JInv.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
  JInv.block(3, 3, 3, 3) = SO3d::hat(T.so3().log());
  JInv = JInv * 0.5 + Matrix6d::Identity();
  return JInv;
}

typedef Matrix<double, 6, 1> Vector6d;


class PoseGraphVertex: public g2o::BaseVertex<6, SE3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   virtual void setToOriginImpl() override
   {
     _estimate = SE3d();
   }

    virtual void oplusImpl(const double* update) override
    {
      Vector6d updateVec;
      updateVec<<update[0], update[1], update[2], update[3], update[4], update[5];
      _estimate = SE3d::exp(updateVec) * _estimate;
    }

    virtual bool read(istream &is) override{
      double data[7];
      for(int i=0; i < 7; ++i)
      {
        is >> data[i];
      }

      setEstimate(SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                      Vector3d(data[0], data[1], data[2])));
      return true;
    }

    virtual bool write(ostream &os) const override
    {
      os << id()<<" ";
      Quaterniond q = _estimate.unit_quaternion();
      os<<_estimate.translation().transpose()<<" ";
      os<<q.coeffs()[0]<<" "<<q.coeffs()[1]<<" "<<q.coeffs()[2]<<" "<<q.coeffs()[3]<<endl;
      return true;
    }
};

class PoseGraphEdge: public g2o::BaseBinaryEdge<6, SE3d,  PoseGraphVertex, PoseGraphVertex>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
      virtual bool read(istream &is) override
      {
        double data[7];
        for(int i=0; i < 7; ++i)
        {
          is >>data[i];
        }
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(SE3d(q, Vector3d(data[0], data[1], data[2])));

        for(int i=0; i < information().rows() && is.good(); ++i)
        {
          for(int j=i; j < information().cols() && is.good(); ++j)
          {
            is >> information()(i, j);
            if(i != j)
            {
              information()(j, i) = information()(i, j);
            }
          }
        }
        return true;
      }

    virtual bool write(ostream &os) const override
    {
      Quaterniond q = _measurement.unit_quaternion();
      PoseGraphVertex *v1 = static_cast<PoseGraphVertex *>(_vertices[0]);
      PoseGraphVertex *v2 = static_cast<PoseGraphVertex *>(_vertices[1]);
      os<<v1->id()<<" "<<v2->id()<<" ";
      os<<_measurement.translation().transpose()<<" ";
      os<<q.coeffs()[0]<<" "<<q.coeffs()[1]<<" "<<q.coeffs()[2]<<" "<<q.coeffs()[3]<<" ";

      for(int i=0; i < information().rows(); ++i)
      {
        for(int j=i; j < information().rows(); ++j)
        {
          os<<information()(i, j)<<" ";
        }
      }
      os<<endl;
      return true;
    }

    virtual void computeError() override
    {
      SE3d v1 = (static_cast<PoseGraphVertex*>(_vertices[0]))->estimate();
      SE3d v2 = (static_cast<PoseGraphVertex*>(_vertices[1]))->estimate();
      _error = ( _measurement.inverse() * v1.inverse() *v2).log();
    }

    virtual void linearizeOplus() override
    {
      auto JInv = computeJInv(SE3d::exp(_error));
      SE3d v1 = (static_cast<PoseGraphVertex*>(_vertices[0]))->estimate();
      SE3d v2 = (static_cast<PoseGraphVertex*>(_vertices[1]))->estimate();
      _jacobianOplusXi = - JInv * v2.inverse().Adj();
      _jacobianOplusXj = JInv * v2.inverse().Adj();
    }
};




int main(int argc, char** argv)
{
  ifstream fin(argv[1]);
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
  typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  int vertexCnt=0, edgeCnt=0;

  vector<PoseGraphVertex* > vertices;
  vector<PoseGraphEdge*> edges;

  while(!fin.eof())
  {
    string name;
    fin >> name;
    if(name == "VERTEX_SE3:QUAT")
    {
      PoseGraphVertex* v = new PoseGraphVertex();
      int index;
      fin >> index;
      v->setId(index);
      v->read(fin);
      optimizer.addVertex(v);
      ++vertexCnt;
      vertices.push_back(v);
      if(index == 0)
      {
        v->setFixed(true);
      }

    }
    else if(name == "EDGE_SE3:QUAT")
    {
      PoseGraphEdge* e = new PoseGraphEdge();
      int idx1, idx2;
      fin>> idx1 >> idx2;
      e->setId(edgeCnt++);
      e->setVertex(0, optimizer.vertices()[idx1]);
      e->setVertex(1, optimizer.vertices()[idx2]);
      e->read(fin);
      optimizer.addEdge(e);
      edges.push_back(e);
    }

    if(!fin.good()) break;
  }

  cout<<"read "<<vertexCnt<<" vertices,"<<edgeCnt<<" edges."<<endl;
  cout<<"start optimization"<<endl;
  optimizer.initializeOptimization();
  optimizer.optimize(30);
  cout<<"saving optimizaiton results ..."<<endl;

  ofstream fout("results_lie.g2o");
  for(PoseGraphVertex *v: vertices)
  {

  fout<<"VERTEX_SE3:QUAT ";
    v->write(fout);
  }


  for(PoseGraphEdge *e:edges)
  {
  fout<<"EDGE_SE3:QUAT ";
  e->write(fout);
  }

  fout.close();
  return 0;

}
