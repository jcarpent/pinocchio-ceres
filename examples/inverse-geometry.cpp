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

#include <pinocchio/spatial/explog.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <pinocchio/parsers/urdf.hpp>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

struct PlacementTask : ceres::CostFunction
{
  typedef double Scalar;
  
  typedef se3::SE3Tpl<Scalar> SE3;
  typedef se3::MotionTpl<Scalar> Motion;
  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> ConfigVectorType;
  typedef Eigen::Matrix<Scalar,6,1> ResidualVectorType;
  typedef se3::Data::Matrix6x JacobianType;
  typedef Eigen::Matrix<double,6,6> Matrix6;
  
  PlacementTask(const se3::Model & model,
                const se3::SE3 & Mee_ref,
                const se3::Model::JointIndex ee_id)
  : model(model)
  , own_data(model)
  , Mee_ref(Mee_ref)
  , ee_id(ee_id)
  , q(model.nq)
  , jacobian_ee(JacobianType::Zero(6,model.nv))
  , jacobian_residual(JacobianType::Zero(6,model.nv))
  {
    mutable_parameter_block_sizes()->push_back(model.nq);
    set_num_residuals(6);
  }

  template<typename ConfigVectorType, typename ResidualVector, typename JacobianType>
  bool Evaluate(const Eigen::MatrixBase<ConfigVectorType> & q,
                const Eigen::MatrixBase<ResidualVector> & residual,
                const Eigen::MatrixBase<JacobianType> & jacobian) const
  {
    typedef Motion::Vector6 Vector6;
    typedef Eigen::Map<Vector6> MapVector6;
    
    se3::MotionRef<ResidualVector> v(EIGEN_CONST_CAST(ResidualVector,residual));
    
    // do forward kinematics and compute all Jacobians for all joints
    se3::computeJointJacobians(model, own_data, q);
    
    const SE3 & Mee = own_data.oMi[ee_id];
    const SE3 M_relative = Mee_ref.actInv(Mee);
    v = se3::log6(M_relative);
    
    // Compute the geometric Jacobian of the end-effector
    se3::getJointJacobian(model,own_data,ee_id,se3::LOCAL,jacobian_ee);
    se3::Jlog6(M_relative, Jlog);
    
    EIGEN_CONST_CAST(JacobianType,jacobian).noalias() = Jlog * jacobian_ee;

    return true;
  }
  
  template<typename ConfigVectorType, typename ResidualVector>
  bool Evaluate(const Eigen::MatrixBase<ConfigVectorType> & q,
                const Eigen::MatrixBase<ResidualVector> & residual) const
  {
    typedef Motion::Vector6 Vector6;
    typedef Eigen::Map<Vector6> MapVector6;
    
    se3::MotionRef<ResidualVector> v(EIGEN_CONST_CAST(ResidualVector,residual));
    
    // do forward kinematics over all joints
    se3::forwardKinematics(model, own_data, q);
    
    const SE3 & Mee = own_data.oMi[ee_id];
    const SE3 M_relative = Mee_ref.actInv(Mee);
    v = se3::log6(M_relative);
    
    return true;
  }
  
  virtual bool Evaluate(double const* const* x,
                        double* residuals,
                        double** jacobians) const
  {
    assert(x != NULL && "x is NULL");
    
    typedef Motion::Vector6 Vector6;
    typedef Eigen::Map<Vector6> MapVector6;
    
    const Eigen::Map<const ConfigVectorType> q_map(x[0],model.nq,1);
    q = q_map;
    
    if(jacobians)
    {
      Evaluate(q,res,jacobian_residual);
      Eigen::Map<ResidualVectorType> residuals_map(residuals,6,1);
      residuals_map = res;
      Eigen::Map<EIGEN_PLAIN_ROW_MAJOR_TYPE(JacobianType)>(jacobians[0],6,model.nv)
      = jacobian_residual;
    }
    else
    {
      Evaluate(q,res);
      Eigen::Map<ResidualVectorType> residuals_map(residuals,6,1);
      residuals_map = res;
    }

    return true;
  }
  
  // data
  const se3::Model & model;
  mutable se3::Data own_data;
  const SE3 Mee_ref;
  const se3::Model::JointIndex ee_id;
  
  mutable Matrix6 Jlog;
  mutable ResidualVectorType res;
  mutable se3::Data::ConfigVectorType q;
  mutable se3::Data::Matrix6x jacobian_ee, jacobian_residual;
                               
};

int main(int /*argc*/, char** argv)
{
  // Init random seed
  std::srand((unsigned int) std::time(0));
  
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
  
  // Init logs
  google::InitGoogleLogging(argv[0]);
  
  // Load the robot model from URDF
  const std::string & filename = MODEL_DIRECTORY"/lwr-robot-description/lwr-robot.urdf";
  Model model; // empty model
  std::cout << "Opening model: " << filename << std::endl;
  se3::urdf::buildModel(filename, model);
  Data data(model);
  
  // Select random configuration
  ConfigVectorType q_random = randomConfiguration(model);
  forwardKinematics(model, data, q_random);
  Model::JointIndex ee_id = model.joints.size()-1;
  const SE3 Mee_ref = data.oMi[ee_id]; // reference placement of the end effector
  
  // The variable to solve for with its initial value.
  const ConfigVectorType q_init(ConfigVectorType::Zero(model.nq));
  ConfigVectorType q_optimization = q_init;
  
  // Build the problem.
  Problem problem;
  
  PlacementTask * placement_task = new PlacementTask(model,Mee_ref,ee_id);
  problem.AddParameterBlock(q_optimization.data(), (int)q_optimization.size());
  problem.AddResidualBlock(placement_task, NULL, q_optimization.data());
  
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
  options.function_tolerance = 1e-10;
  options.check_gradients = true;
  options.gradient_check_numeric_derivative_relative_step_size = 1e-8;
  options.gradient_check_relative_precision = 2;
//  options.gradient_tolerance = 1e-4;
  
  std::vector<double*> parameter_blocks;
  parameter_blocks.push_back(q_optimization.data());
  
  ceres::NumericDiffOptions numeric_diff_options;
  numeric_diff_options.relative_step_size = 1e-8;
  ceres::GradientChecker gradient_checker(placement_task,
                                          NULL, numeric_diff_options);
  ceres::GradientChecker::ProbeResults results;
  gradient_checker.Probe(parameter_blocks.data(), std::sqrt(numeric_diff_options.relative_step_size), &results);
  std::cout << "gradient_checker log:" << results.error_log << std::endl;

  Solver::Summary summary;
  Solve(options, &problem, &summary);
  
  // Compute final error
  forwardKinematics(model, data, q_optimization);
  const SE3 & Mee_final = data.oMi[ee_id];
  Motion error_final = se3::log6(Mee_final.actInv(Mee_ref));
  
  std::cout << summary.FullReport() << "\n";
  std::cout << "initial configuration:\n" << q_init.transpose() << std::endl;
  std::cout << "optimal configuration:\n" << q_optimization.transpose()  << std::endl;
  std::cout << "random configuration:\n" << q_random.transpose()  << std::endl;
  std::cout << "relative final error: " << error_final.toVector().norm() << std::endl;
  
  return 0;
}
