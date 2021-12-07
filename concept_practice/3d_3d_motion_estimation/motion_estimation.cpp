#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "feature_matching.hpp"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;

typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Vectors3d;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> Vectors2d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;



cv::Point2d pixel2Cam(const cv::Point2d& point, const cv::Mat& K)
{
  cv::Point2d normalisedPoint;
  double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), px = K.at<double>(0,2),
         py = K.at<double>(1, 2);
  normalisedPoint.x = (point.x - px) / fx;
  normalisedPoint.y = (point.y - py) / fy;
  return normalisedPoint;
}

void triangulate(const vector<cv::KeyPoint>& kp1, const vector<cv::KeyPoint>& kp2,
                 const vector<cv::DMatch> &matches, const cv::Mat &R, const cv::Mat &t,
                 cv::Mat &K, vector<cv::Point3d>& points3D)
{
  cv::Mat T1 = (cv::Mat_<double>(3, 4)<<
           1.0, 0, 0, 0,
           0, 1.0, 0, 0,
           0, 0, 1.0, 0);
  cv::Mat T2 = (cv::Mat_<double>(3, 4)<<
           R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
           R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
           R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
  cout<<"T value"<<T2.at<double>(0, 3)<<endl<<T2.at<double>(1, 3)<<endl<<T2.at<double>(2, 3)<<endl<<T2.at<double>(0,0)<<endl;
  vector<cv::Point2d> points1, points2;

  for(auto& match: matches)
  {
    auto normalised_point1 = pixel2Cam(kp1[match.queryIdx].pt, K);
    auto normalised_point2 = pixel2Cam(kp2[match.trainIdx].pt, K);

    points1.emplace_back(normalised_point1.x, normalised_point1.y);
    points2.emplace_back(normalised_point2.x, normalised_point2.y);
  }

  cv::Mat points_4d;

  cv::triangulatePoints(T1, T2, points1, points2, points_4d);

  cout<<"printing points"<<endl;
  //for(cv::Point3d& point:points4D)
  for(int i=0; i < points_4d.cols; ++i)
  {
    cv::Mat point = points_4d.col(i);
    point /= point.at<double>(3, 0);
    points3D.emplace_back(point.at<double>(0, 0), point.at<double>(1, 0),
                         point.at<double>(2, 0));
    cout<<point.at<double>(0, 0)<<" "<<point.at<double>(1, 0)<<" "<<point.at<double>(2, 0)
        <<endl;
  }

}


void solvePnPManual(const Vectors3d& points_3d, const Vectors2d& points_2d,
                    const cv::Mat& K, Sophus::SE3d &pose)
{
  int maxIterations = 10;

  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  double cost=0, lastCost=0;

  


  for(int i=0; i < maxIterations; ++i)
  {

    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d rhs = Vector6d::Zero();
    Vector6d dx = Vector6d::Zero();
    cost = 0;

    for (int j=0; j < points_3d.size(); ++j)
    {

      Eigen::Matrix<double, 2, 6> J = Eigen::Matrix<double, 2, 6>::Zero();
      Eigen::Vector3d point = pose * points_3d[j];
      Eigen::Vector2d error = Eigen::Vector2d::Zero();
      double u = fx * point[0] / point[2] + cx;
      double v = fy * point[1] / point[2] + cy;
      error[0] = points_2d[j][0] - u;
      error[1] = points_2d[j][1] - v;
      cost += error[0] * error[0] + error[1] * error[1];

      double zSquared = point[2] * point[2];
      double invZ = 1.0 / point[2];
      double invZSquare = invZ * invZ;
      J(0, 0) += -fx * invZ;
      J(0, 2) += fx * point[0] * invZSquare;
      J(0, 3) += fx * point[0] * point[1] * invZSquare;
      J(0, 4) += -fx - fx * point[0] * point[0] * invZSquare;
      J(0, 5) += fx * point[1] * invZ;
      J(1, 1) += -fy * invZ;
      J(1, 2) += fy * point[1] * invZSquare;
      J(1, 3) += fy + fy * point[1] * point[1] * invZSquare;
      J(1, 4) += -fy * point[0] * point[1] * invZSquare;
      J(1, 5) += -fy * point[0] * invZ;

      rhs += -J.transpose() * error;
      H += J.transpose() * J;
    }

    dx = H.ldlt().solve(rhs);
    if(isnan(dx[0]))
    {
      cout<<"result is nan"<<endl;
      break;
    }


    if(i > 0 && cost >= lastCost)
    {
      cout<<"cost increasing "<<cost<<endl;
      break;
    }

    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;
    cout<<"iteration: "<<i<<" cost="<<cost<<endl;
    if(dx.norm() < 1e-6)
    {
      break;
    }


  }

  cout<<"converged pose: "<<pose.matrix()<<endl;

}


// Start of g2o related stufss
//
class VertexPose: public g2o::BaseVertex<6, Sophus::SE3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override{
      _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double* update) override{
       Vector6d updateEigen;
       updateEigen<< update[0], update[1], update[2], update[3], update[4], update[5];
       _estimate = Sophus::SE3d::exp(updateEigen) * _estimate;
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}
};

class EdgeProjection: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K): _pos3d(pos), _K(K){}

    virtual void computeError() override{
      const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
      Sophus::SE3d T = v->estimate();
      Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
      pos_pixel /= pos_pixel[2];
      _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override
    {
      const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);

      Sophus::SE3d T = v->estimate();
      Eigen::Vector3d posCam = T * _pos3d;

      double fx =  _K(0, 0);
      double fy =  _K(1, 1);
      double px = _K(0, 2);
      double py = _K(1, 2);
      double X = posCam[0];
      double Y = posCam[1];
      double Z = posCam[2];
      double ZInv = 1.0 / Z;
      double Z2Inv = ZInv * ZInv;
      double Z2 = Z * Z;

      _jacobianOplusXi
        <<-fx *  ZInv, 0.0, fx * X * Z2, fx * X * Y * Z2Inv, - fx - fx * X * X * Z2Inv, -fx * Y * ZInv,
          0.0, -fy * ZInv, fy * Y * Z2Inv, fy + fy * Y*Y * Z2Inv, -fy * X * Y * Z2Inv, -fy * X * ZInv;
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}


    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;

};


void solvePnPusingG2O(
    const Vectors3d &points3D,
    const Vectors2d &points2D,
    const cv::Mat &K,
    Sophus::SE3d &pose
    )
{
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
      );

  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);


  int index=0;
  VertexPose *vertexPose = new VertexPose();
  vertexPose->setId(index);
  vertexPose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertexPose);
  ++index;


  Eigen::Matrix3d KEigen;
  KEigen<< 
    K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);


  for(size_t i=0; i < points3D.size(); ++i)
  {
    auto p3d = points3D[i];
    auto p2d = points2D[i];

    EdgeProjection *edge = new EdgeProjection(p3d, KEigen);
    edge->setId(0);
    edge->setVertex(0, vertexPose);
    edge->setMeasurement(p2d);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }


  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout<<"time used by g2o: "<<time.count()<<" seconds"<<endl;
  cout<<"final pose "<<vertexPose->estimate().matrix()<<endl;
  pose = vertexPose->estimate();
}


void solveICPusingSVD(
    Vectors3d &points1_3D,
    Vectors3d &points2_3D,
    Eigen::Matrix<double, 3, 3>& R,
    Eigen::Vector3d &t
    )
{
  Eigen::Vector3d px, qx;
  Eigen::Matrix<double, 3, 3> W;
  px = Eigen::Vector3d::Zero();
  qx = Eigen::Vector3d::Zero();
  W = Eigen::Matrix<double, 3, 3>::Zero();


  for(int i=0; i < points1_3D.size(); ++i)
  {
    px += points1_3D[i];
    qx += points2_3D[i];
  }
  px /= points1_3D.size();
  qx /= points2_3D.size();

  for(int i=0; i < points2_3D.size(); ++i)
  {
    points1_3D[i] -= px;
    points2_3D[i] -= qx;
    W += points1_3D[i] * points2_3D[i].transpose();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  R = svd.matrixU() * svd.matrixV().transpose();
  if(R.determinant() < 0)
  {
    R = -R;
  }

  t = px - R * qx;
}




int main()
{
  const cv::Mat img1 = cv::imread("1.png", 0);
  const cv::Mat img2 = cv::imread("2.png", 0);
  const cv::Mat depth1 = cv::imread("1_depth.png", cv::IMREAD_UNCHANGED);
  const cv::Mat depth2 = cv::imread("2_depth.png", cv::IMREAD_UNCHANGED);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<cv::DMatch> matches;
  vector<cv::KeyPoint> kp1, kp2;

  match_image_features(img1, img2, matches, kp1, kp2);
  vector<cv::Point3d> points1_3d, points2_3d;
  vector<cv::Point2d> points_2d;

  for(const auto& match: matches)
  {
    ushort depth = depth1.at<unsigned short>(int(kp1[match.queryIdx].pt.y), int(kp1[match.queryIdx].pt.x));
    ushort depth2 = depth1.at<unsigned short>(int(kp2[match.trainIdx].pt.y), int(kp2[match.trainIdx].pt.x));
    if(depth == 0 || depth2 == 0)
      continue;
    double scaledDepth = depth / 5000.0;

    cv::Point2d normalisedPoint = pixel2Cam(kp1[match.queryIdx].pt, K);
   points1_3d.push_back(cv::Point3d(normalisedPoint.x * scaledDepth,
                                   normalisedPoint.y * scaledDepth,
                                   scaledDepth));
   points_2d.emplace_back(kp2[match.trainIdx].pt.x, kp2[match.trainIdx].pt.y);
   cv::Point2d normalisedPoint2 = pixel2Cam(kp2[match.trainIdx].pt, K);
   double scaledDepth2 = depth2 / 5000.0;
   points2_3d.push_back(cv::Point3d(normalisedPoint2.x * scaledDepth2,
                                    normalisedPoint2.y * scaledDepth2,
                                    scaledDepth2));
  }

  cv::Mat r,t;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  cv::solvePnP(points1_3d, points_2d, K, cv::Mat(), r, t, false, cv::SOLVEPNP_ITERATIVE);
  cv::Mat R;
  cv::Rodrigues(r, R);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout<<" time for solve pnp: "<<time_used.count()<<" seconds"<<endl;
  cout<<"R = "<<endl<<R<<endl;
  cout<<"t="<<endl<<t<<endl;


  Vectors3d eigen_vectors_3d_pts1, eigen_vectors_3d_pts2;
  Vectors2d eigen_vectors_2d;
  for(int i=0; i < points1_3d.size(); ++i)
  {
    eigen_vectors_3d_pts1.push_back(Eigen::Vector3d(points1_3d[i].x, points1_3d[i].y, points1_3d[i].z));
    eigen_vectors_2d.push_back(Eigen::Vector2d(points_2d[i].x, points_2d[i].y));
    eigen_vectors_3d_pts2.push_back(Eigen::Vector3d(points2_3d[i].x, points2_3d[i].y, points2_3d[i].z));
  }

  Sophus::SE3d pose;

  chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
  solvePnPManual(eigen_vectors_3d_pts1, eigen_vectors_2d, K, pose);
  chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
  chrono::duration<double> time_used2 = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
  cout<<"time used manual pnp: "<<time_used2.count()<<" seconds"<<endl;

  Sophus::SE3d newPose;
  solvePnPusingG2O(eigen_vectors_3d_pts1, eigen_vectors_2d, K, newPose);

  Eigen::Matrix<double, 3, 3> R1;
  Eigen::Vector3d tEigen;
  chrono::steady_clock::time_point t5 = chrono::steady_clock::now();
  solveICPusingSVD(eigen_vectors_3d_pts1, eigen_vectors_3d_pts2, R1, tEigen);
  chrono::steady_clock::time_point t6 = chrono::steady_clock::now();
  chrono::duration<double> time_used3 = chrono::duration_cast<chrono::duration<double>>(t6 - t5);
  cout<<"time used for ICP: "<<time_used3.count()<<" seconds"<<endl;
  cout<<R1<<endl;
  cout<<tEigen.transpose()<<endl;
  
  cout<<"Alternate soltion"<<endl;
  cout<<R1.transpose()<<endl;
  cout<<(-R1.transpose()*tEigen).transpose()<<endl;

  //for(int i=0; i < point)
  for(int i=0; i < eigen_vectors_3d_pts1.size(); ++i)
  {
    cout<<"point 2"<<endl;
    cout<<eigen_vectors_3d_pts2[i].transpose()<<endl;
    cout<<"point Rp1+t"<<endl;
    cout<< (R1 * eigen_vectors_3d_pts1[i] + tEigen).transpose()<<endl;
  }

  return 0;
}
