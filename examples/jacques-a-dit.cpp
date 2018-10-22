// 
// Copyright (c) 2018 INRIA
//
// This file is part of Pinocchio-Ceres
// pinocchio is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
// pinocchio is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Lesser Public License for more details. You should have
// received a copy of the GNU Lesser General Public License along with
// pinocchio If not, see
// <http://www.gnu.org/licenses/>.
//

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "ceres/gradient_checker.h"

#include <pinocchio/ceres/local-parameterization.hpp>

#include <pinocchio/spatial/explog.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <pinocchio/parsers/urdf.hpp>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

struct ProximalConfigurationCost : ceres::CostFunction
{
  typedef double Scalar;
  enum { Options = 0 };
  
  typedef se3::ModelTpl<Scalar,Options> Model;
  typedef se3::DataTpl<Scalar,Options> Data;
  
  typedef se3::SE3Tpl<Scalar,Options> SE3;
  typedef se3::MotionTpl<Scalar,Options> Motion;
  typedef Model::ConfigVectorType ConfigVectorType;
  typedef Model::TangentVectorType TangentVectorType;
  typedef TangentVectorType ResidualVectorType;
  typedef Data::MatrixXs JacobianType;
  
  template<typename ConfigVectorLike>
  ProximalConfigurationCost(const se3::Model & model,
                            const Eigen::MatrixBase<ConfigVectorLike> & q_ref)
  : model(model)
  , q(model.nq)
  , q_ref(q_ref)
  , jacobian_residual(JacobianType::Zero(model.nv,model.nv))
  {
    mutable_parameter_block_sizes()->push_back(model.nq);
    set_num_residuals(model.nv);
    jacobian_expressed_relatively_to_the_tangent_ = true;
  }
  
  template<typename ConfigVectorType, typename ResidualVector, typename JacobianType>
  bool Evaluate(const Eigen::MatrixBase<ConfigVectorType> & q,
                const Eigen::MatrixBase<ResidualVector> & residual,
                const Eigen::MatrixBase<JacobianType> & jacobian) const
  {
    EIGEN_CONST_CAST(JacobianType,jacobian).diagonal().setOnes();
    EIGEN_CONST_CAST(ResidualVector,residual) = -se3::difference(model,q,q_ref);
    
    return true;
  }
  
  template<typename ConfigVectorType, typename ResidualVector>
  bool Evaluate(const Eigen::MatrixBase<ConfigVectorType> & q,
                const Eigen::MatrixBase<ResidualVector> & residual) const
  {
    // just call rnea
    EIGEN_CONST_CAST(ResidualVector,residual) = -se3::difference(model,q,q_ref);
    
    return true;
  }
  
  virtual bool Evaluate(double const* const* x,
                        double* residuals,
                        double** jacobians) const
  {
    assert(x != NULL && "x is NULL");
    
    const Eigen::Map<const ConfigVectorType> q_map(x[0],model.nq,1);
    q = q_map;
    
    if(jacobians)
    {
      Evaluate(q,res,jacobian_residual);
      Eigen::Map<ResidualVectorType> residuals_map(residuals,model.nv,1);
      residuals_map = res;
      
      Eigen::Map<EIGEN_PLAIN_ROW_MAJOR_TYPE(JacobianType)>(jacobians[0],model.nv,model.nq).leftCols(model.nv)
      = jacobian_residual;
      //      std::cout << "jacobian_residual\n" << jacobian_residual << std::endl;
    }
    else
    {
      Evaluate(q,res);
      Eigen::Map<ResidualVectorType> residuals_map(residuals,model.nv,1);
      residuals_map = res;
    }
    
    return true;
  }
  
  // data
  const se3::Model & model;
  
  mutable ResidualVectorType res;
  mutable ConfigVectorType q, q_ref;
  mutable JacobianType jacobian_residual;
  
};

struct GravityCompensationTask : ceres::CostFunction
{
  typedef double Scalar;
  enum { Options = 0 };
  
  typedef se3::ModelTpl<Scalar,Options> Model;
  typedef se3::DataTpl<Scalar,Options> Data;
  
  typedef se3::SE3Tpl<Scalar,Options> SE3;
  typedef se3::MotionTpl<Scalar,Options> Motion;
  typedef Model::ConfigVectorType ConfigVectorType;
  typedef Model::TangentVectorType TangentVectorType;
  typedef TangentVectorType ResidualVectorType;
  typedef Data::MatrixXs JacobianType;

  GravityCompensationTask(const se3::Model & model)
  : model(model)
  , own_data(model)
  , q(model.nq)
  , jacobian_residual(JacobianType::Zero(model.nv,model.nv))
  {
    mutable_parameter_block_sizes()->push_back(model.nq);
    set_num_residuals(model.nv);
    jacobian_expressed_relatively_to_the_tangent_ = true;
  }

  template<typename ConfigVectorType, typename ResidualVector, typename JacobianType>
  bool Evaluate(const Eigen::MatrixBase<ConfigVectorType> & q,
                const Eigen::MatrixBase<ResidualVector> & residual,
                const Eigen::MatrixBase<JacobianType> & jacobian) const
  {
    se3::computeGeneralizedGravityDerivatives(model,own_data,q,EIGEN_CONST_CAST(JacobianType,jacobian));
    EIGEN_CONST_CAST(ResidualVector,residual) = own_data.g;

    return true;
  }
  
  template<typename ConfigVectorType, typename ResidualVector>
  bool Evaluate(const Eigen::MatrixBase<ConfigVectorType> & q,
                const Eigen::MatrixBase<ResidualVector> & residual) const
  {
    // just call rnea
    EIGEN_CONST_CAST(ResidualVector,residual)
    = se3::computeGeneralizedGravity(model, own_data, q);
    
    return true;
  }
  
  virtual bool Evaluate(double const* const* x,
                        double* residuals,
                        double** jacobians) const
  {
    assert(x != NULL && "x is NULL");
    
    const Eigen::Map<const ConfigVectorType> q_map(x[0],model.nq,1);
    q = q_map;
    
    if(jacobians)
    {
      Evaluate(q,res,jacobian_residual);
      Eigen::Map<ResidualVectorType> residuals_map(residuals,model.nv,1);
      residuals_map = res;
      
      Eigen::Map<EIGEN_PLAIN_ROW_MAJOR_TYPE(JacobianType)>(jacobians[0],model.nv,model.nq).leftCols(model.nv)
      = jacobian_residual;
//      std::cout << "jacobian_residual\n" << jacobian_residual << std::endl;
    }
    else
    {
      Evaluate(q,res);
      Eigen::Map<ResidualVectorType> residuals_map(residuals,model.nv,1);
      residuals_map = res;
    }

    return true;
  }
  
  // data
  const se3::Model & model;
  mutable se3::Data own_data;
  
  mutable ResidualVectorType res;
  mutable ConfigVectorType q;
  mutable JacobianType jacobian_residual;
                               
};

int main(int /*argc*/, char** argv)
{
  // Init random seed
  srand((unsigned int) time(0));
  
  using ceres::AutoDiffCostFunction;
  using ceres::NumericDiffCostFunction;
  using ceres::CostFunction;
  using ceres::Problem;
  using ceres::Solver;
  using ceres::Solve;
  
  using namespace se3;
  
  typedef double Scalar;
  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> ConfigVectorType;
  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> TangentVectorType;
  
  bool with_floating_base = false;
//  with_floating_base = true;
  
  // Init logs
  google::InitGoogleLogging(argv[0]);
  
  // Load the robot model from URDF
  const std::string & filename = MODEL_DIRECTORY"/lwr-robot-description/lwr-robot.urdf";
  Model model; // empty model
  
  std::cout << "Opening model: " << filename << std::endl;
  if(with_floating_base)
  {
    se3::urdf::buildModel(filename, se3::JointModelFreeFlyer(), model);
    model.lowerPositionLimit.head<7>().fill(-1.1);
    model.upperPositionLimit.head<7>().fill( 1.1);
  }
  else
    se3::urdf::buildModel(filename, model);
  
  std::cout << "model.nq: " << model.nq << "; model.nv: " << model.nv << std::endl;
  
  Data data(model);
  
  // Define local parametrization for the model
  typedef pinocchio::ceres::ModelLocalParameterization<Model::Scalar,Model::Options> ModelLocalParameterization;
  ModelLocalParameterization * local_para_ptr = new ModelLocalParameterization(model);

  // The variable to solve for with its initial value.
  std::cout << "lb: " << model.lowerPositionLimit.transpose() << std::endl;
  std::cout << "ub: " << model.upperPositionLimit.transpose() << std::endl;
  
  {
    ConfigVectorType q_random = randomConfiguration(model);
  }
  ConfigVectorType q_random = randomConfiguration(model); // random initial value
  std::cout << "q_random: " << q_random.transpose() << std::endl;
  
  ConfigVectorType q_optimization = q_random;
  
  // Build the problem.
  Problem problem;
  
  GravityCompensationTask * main_task = new GravityCompensationTask(model);
  const ConfigVectorType q_ref = ConfigVectorType::Zero(model.nq);
  ProximalConfigurationCost * config_cost = new ProximalConfigurationCost(model,q_ref);
  // Evaluate at initial point both the residual and the Jacobian
  GravityCompensationTask::ResidualVectorType res_init(model.nv,1);
  GravityCompensationTask::JacobianType jac_init(model.nv,model.nv);
  main_task->Evaluate(q_random,res_init,jac_init);
//  std::cout << "res_init: " << res_init.transpose() << std::endl;
//  std::cout << "jac_init:\n" << jac_init<< std::endl;
  

  problem.AddParameterBlock(q_optimization.data(), (int)q_optimization.size(), local_para_ptr);
  problem.AddResidualBlock(main_task, NULL, q_optimization.data());

  for(int k = 0; k < model.nq; ++k)
  {
    problem.SetParameterLowerBound(q_optimization.data(), k, model.lowerPositionLimit[k]);
    problem.SetParameterUpperBound(q_optimization.data(), k, model.upperPositionLimit[k]);
  }

  // Run the solver!
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 500;
  options.function_tolerance = 1e-14;
  options.check_gradients = true;
  options.parameter_tolerance = 1e-12;
  options.gradient_check_numeric_derivative_relative_step_size = 1e-8;
  options.gradient_check_relative_precision = 2;
  options.trust_region_strategy_type = ceres::DOGLEG;
//  options.dogleg_type = ceres::SUBSPACE_DOGLEG;
//  options.logging_type = ceres::PER_MINIMIZER_ITERATION;
//  options.gradient_tolerance = 1e-4;
  
  
  std::vector<double*> parameter_blocks;
  parameter_blocks.push_back(q_optimization.data());
  
  ceres::NumericDiffOptions numeric_diff_options;
  numeric_diff_options.relative_step_size = 1e-8;

  std::vector<const ceres::LocalParameterization*> problem_local_para_vector;
  problem_local_para_vector.push_back(local_para_ptr);
  ceres::GradientChecker gradient_checker(/*main_task*/ config_cost,
                                          &problem_local_para_vector, numeric_diff_options);
  ceres::GradientChecker::ProbeResults results;
  gradient_checker.Probe(parameter_blocks.data(), 1e3*std::sqrt(numeric_diff_options.relative_step_size), &results);
  std::cout << "gradient_checker log:" << results.error_log << std::endl;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  // Compute final error
  GravityCompensationTask::ResidualVectorType residual_final;
  main_task->Evaluate(q_optimization, residual_final);

  std::cout << summary.FullReport() << "\n";
//  std::cout << "optimal configuration:\n" << q_optimization.transpose()  << std::endl;
  std::cout << "random initial configuration:\n" << q_random.transpose()  << std::endl;
  std::cout << "final residual: " << residual_final.squaredNorm() << std::endl;
  std::cout << "optimal configuration:\n" << q_optimization.transpose()  << std::endl;
  std::cout << "distance to lower bound:\n" << se3::difference(model,q_optimization,model.lowerPositionLimit).transpose() << std::endl;
  std::cout << "distance to upper bound:\n" << se3::difference(model,q_optimization,model.upperPositionLimit).transpose() << std::endl;

  std::cout << "final norm: " << q_optimization.segment<4>(3).norm() << std::endl;
  return 0;
}
