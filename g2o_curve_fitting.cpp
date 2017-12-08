/*
  
 模仿高博的《视觉SLAM十四讲》里第六讲的g2o测试程序
 
 目标函数：
 y1 = a * x^3 + b * x^2 + c * x;
 y2 = a * x^5 + b * x^4 + c * x^3;
 
 噪声：
 sigma_y1加在测量值y1上，为均值为0，方差为0.01的高斯噪声
 sigma_y2加在测量值y2上，为均值为0，方差为0.02的高斯噪声
 
 优化abc三个参数
 
 */

//基本头文件
#include <iostream>
#include <cmath>
#include <chrono>
//opencv和eigen
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Core>
//包含边的头文件
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
//求解器块头文件
#include <g2o/core/block_solver.h>
//求解算法的头文件
#include <g2o/core/optimization_algorithm_levenberg.h>
//求解器类型头文件
#include <g2o/solvers/dense/linear_solver_dense.h>
//噪声发生器
#include <g2o/stuff/sampler.h>

using namespace std;
using namespace cv;
using namespace g2o;

//定义顶点，即要优化的对象
//因为我们要优化a，b，c三个参数，所以顶点为三维，数据类型为vector3d
//同时因为要优化一组abc，所以顶点也只有一个，这在主函数中也有所体现
//3是顶点维度，Vector3d是顶点类型
class CurveFittingVertex: public BaseVertex<3,Eigen::Vector3d>
{
public:
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  //要优化变量abc的初值设置函数
  virtual void setToOriginImpl()
  {
    _estimate << 0,0,0;
  }
  
  //这是优化变量的更新函数，最小二乘不就是不断加一个极小的变化量么
  virtual void oplusImpl( const double* update )
  {
    _estimate += Eigen::Vector3d(update);
  }
  
  //存读函数，留空
  //这两个函数可能是用来直接获取数据的吧，因为在主程序中有数据（顶点）写入的过程，所以此处留空
  virtual bool read( istream& in ) {}
  virtual bool write( ostream& out ) const {}

};

//定义边，即误差项
//因为方程是两个，也就是说每一组数据（变量）都会产生两组测量（误差），因而是二维
//主函数中会有100组数据，所以也会有100个这样的边，而且都连接到唯一的一个顶点上
//2是边的维度，Vector2d是边的类型，CurveFittingVertex是边所连接的顶点类型，因为是一元边，所以只有一个顶点
class CurveFittingEdge : public BaseUnaryEdge<2,Eigen::Vector2d,CurveFittingVertex>
{
public:
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  //构造函数
  CurveFittingEdge( double x ) : BaseUnaryEdge(),_x(x) {}
  
  //误差计算函数
  void computeError()
  {
    //获取顶点的值
    //在二元边甚至多元边中，会有更多的顶点：_vertices[i]
    const CurveFittingVertex* vertex = static_cast<const CurveFittingVertex*> (_vertices[0]);
    
    //获取顶点的估计值
    Eigen::Vector3d abc_vertex = vertex->estimate();

    //对误差项进行赋值，因为是两个方程，所以_error和_measurement是二维的
    //注意_error和_measurement是如何访问元素值的，因为与边CurveFittingEdge一样，都是vector2d，所以要用（i，j）来访问
    _error(0,0) = _measurement(0,0) -( abc_vertex(0,0)*_x*_x*_x + abc_vertex(1,0)*_x*_x + abc_vertex(2,0)*_x );
    _error(1,0) = _measurement(1,0) -( abc_vertex(0,0)*_x*_x*_x*_x*_x + abc_vertex(1,0)*_x*_x*_x*_x + abc_vertex(2,0)*_x*_x*_x );
  }
  
  //重写雅格比矩阵
  //高博书中说可以让其自动计算，就没有重写，在这里为更好地理解，重写一下
  virtual void linearizeOplus()
  {
    //因为这个雅格比矩阵与三个变量没有关系，所以其实没有必要读出顶点的估计值
    //要注意前面的符号，雅格比矩阵是由误差公式计算出来的
    _jacobianOplusXi(0,0) = - _x*_x*_x;
    _jacobianOplusXi(0,1) = - _x*_x;
    _jacobianOplusXi(0,2) = - _x;
    _jacobianOplusXi(1,0) = - _x*_x*_x*_x*_x;
    _jacobianOplusXi(1,1) = - _x*_x*_x*_x;
    _jacobianOplusXi(1,2) = - _x*_x*_x;
    
  }
  
  //读存函数。留空
  virtual bool read( istream& in ) {}
  virtual bool write( ostream& out ) const {}

public:
  //这个是输入变量，是数据
  //这个是要我们自己定义的，因为我们要计算误差，测量值直接可以获得，二估计测量值就需要用具体的方程计算了
  double _x;
};

//主函数
int main( int argc, char** argv )
{
  //模拟产生带噪声的数据
  //变量真实值
  double a = 2.0 , b = 3.0 , c = 4.0;
  int N = 100;
  //两个测量的噪声方差及协方差（这里设两个噪声相互独立）
  double sigma_y1 = 0.01;
  double sigma_y2 = 0.02;
  
  //存储模拟数据的容器
  vector<double> x_data ,y1_data ,y2_data;
  for( int i=0 ; i<N ; i++ )
  {
    double x = i/100.0;
    x_data.push_back(x);
    //向测量值y1_data,y2_data中加噪声
    y1_data.push_back( a*x*x*x + b*x*x + c*x + Sampler::gaussRand(0,sigma_y1) );
    y2_data.push_back( a*x*x*x*x*x + b*x*x*x*x + c*x*x*x + Sampler::gaussRand(0,sigma_y2) );
  }
  cout<<"带噪声的数据产生完成..."<<endl;
  
  //g2o求解器的建立
  //矩阵块，优化变量维度为3，误差项维度为2
  typedef BlockSolver< BlockSolverTraits<3,2> > blocksolver;
  //求解器类型为稠密性
  blocksolver::LinearSolverType* linearsolver = new LinearSolverDense<blocksolver::PoseMatrixType> ();
  //矩阵求解器指针
  blocksolver* solver_ptr = new blocksolver(linearsolver);
  //求解器
  OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg( solver_ptr );
  
  //优化器（图模型）的建立
  SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  //打开调试输出
  optimizer.setVerbose(true);
  cout<<"优化器建立完成..."<<endl;
  
  //向优化器（图模型）添加顶点
  CurveFittingVertex* v = new CurveFittingVertex();
  //设置顶点初始值即estimate估计值
  v->setEstimate( Eigen::Vector3d(0,0,0) );
  //设置顶点ID
  v->setId(0);
  //向优化器（图模型）中添加顶点
  optimizer.addVertex(v);
  cout<<"顶点添加完成..."<<endl;
  
  //向优化器（图模型）中添加边（误差项）
  for( int i=0;i<N;i++ )
  {
    CurveFittingEdge* e = new CurveFittingEdge( x_data[i] );
    //设置本条边的ID
    e->setId(i);
    //设置本条边所连接的顶点，第一个参数为边链接的第几个顶点（一元边所以只有一个0），第二个参数为顶点对象
    e->setVertex(0,v);
    //设置边的测量值也就是Y
    //测量值应该是与误差项一个类型的
    Eigen::Vector2d measurement;
    measurement << y1_data[i],y2_data[i];
    e->setMeasurement( measurement );
    
    //设置信息矩阵，即协方差矩阵
    //还不是很明白这个矩阵是一个什么意义
    e->setInformation(Eigen::Matrix2d::Identity());
    //向优化器（图模型）中添加边
    optimizer.addEdge(e);   
  }
  cout<<"边添加完成..."<<endl;
  
  //计时
  chrono::steady_clock::time_point time_begin = chrono::steady_clock::now();
  
  //开始优化
  optimizer.initializeOptimization();
  optimizer.optimize(100);
  
  chrono::steady_clock::time_point time_end = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(time_end-time_begin);
  cout<<"求解完成，共耗时"<<time_used.count()<<" s"<<endl;
  
  cout<<"优化后的参数为 "<<v->estimate().transpose()<<endl;;
  return 0;
}