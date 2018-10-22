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

#include <gtest/gtest.h>

#include "pinocchio/ceres/linear-least-square.hpp"
#include "pinocchio/ceres/proximal-cost.hpp"

namespace
{
  
  TEST(ProximalCost,LinearLeastSquare)
  {
    using namespace pinocchio::ceres;
    using ceres::Problem;
    using ceres::Solver;
    Eigen::Index n = 40, m = 10;
    
    Eigen::MatrixXd A(Eigen::MatrixXd::Random(m,n));
    Eigen::VectorXd b(Eigen::VectorXd::Random(m));
    
    Eigen::VectorXd x0(Eigen::VectorXd::Random(n,1));
    
    LinearLeastSquareResidual * lls_ptr = new LinearLeastSquareResidual(A,b);
    const LinearLeastSquareResidual & lls = *lls_ptr;
    Eigen::VectorXd res0(m,1);
    Eigen::MatrixXd jac0(m,n);
    lls.Evaluate(x0,res0,jac0);
    ASSERT_TRUE(res0.isApprox(A*x0-b));
    ASSERT_TRUE(jac0.isApprox(A));
    
    Eigen::VectorXd x_optimization(x0);
    
    // Build the problem.
    Problem problem;
    problem.AddResidualBlock(lls_ptr, NULL, x_optimization.data());
    
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 500;
    options.function_tolerance = 1e-14;
    options.check_gradients = true;
    options.parameter_tolerance = 1e-12;
    options.gradient_check_numeric_derivative_relative_step_size = 1e-8;
    options.gradient_check_relative_precision = 10;
    
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    
    Eigen::VectorXd res_final(m,1);
    lls.Evaluate(x_optimization,res_final);
    ASSERT_TRUE(res_final.isZero());
  }
  
  TEST(ProximalCost,ProximalLinearLeastSquare)
  {
    using namespace pinocchio::ceres;
    using ceres::Problem;
    using ceres::Solver;
    using ceres::ScaledLoss;
    Eigen::Index n = 40, m = 10;
    
    Eigen::MatrixXd A(Eigen::MatrixXd::Random(m,n));
    Eigen::VectorXd b(Eigen::VectorXd::Random(m));
    
    Eigen::VectorXd x0(Eigen::VectorXd::Random(n,1));
    
    LinearLeastSquareResidual * lls_ptr = new LinearLeastSquareResidual(A,b);
    const LinearLeastSquareResidual & lls = *lls_ptr;
    
    ProximalCost * pc_ptr = new ProximalCost(x0);
    const ProximalCost & pc = *pc_ptr;
    Eigen::VectorXd res0_prox(x0.size(),1);
    Eigen::MatrixXd jac0_prox(n,n);
    pc.Evaluate(x0,res0_prox,jac0_prox);
    ASSERT_TRUE(res0_prox.isZero());
    ASSERT_TRUE(jac0_prox.isIdentity());
    
    
    Eigen::VectorXd x_optimization(x0);
    
    // Build the problem.
    Problem problem;
    problem.AddResidualBlock(lls_ptr, NULL, x_optimization.data());

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 500;
    options.function_tolerance = 1e-14;
    options.check_gradients = true;
    options.parameter_tolerance = 1e-12;
    options.gradient_check_numeric_derivative_relative_step_size = 1e-8;
    options.gradient_check_relative_precision = 10;
    
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    
    const Eigen::VectorXd x_sol1 = x_optimization;
    Eigen::VectorXd res_sol1(m,1);
    lls.Evaluate(x_sol1,res_sol1);
    
    const double scaling_factor = 1e-8;
    ScaledLoss * scaling_ptr = new ScaledLoss(NULL,scaling_factor,ceres::DO_NOT_TAKE_OWNERSHIP);
    problem.AddResidualBlock(pc_ptr, scaling_ptr, x_optimization.data());
    
    x_optimization = x0;
    Solver::Summary summary_prox;
    Solve(options, &problem, &summary_prox);
    
    Eigen::VectorXd res_final(m,1);
    lls.Evaluate(x_optimization,res_final);
    ASSERT_GT(res_final.norm(),res_sol1.norm());
    
    bool prox_has_congerged = false;
    const int max_it = 100;
    x_optimization = x0;
    int it = 0;
    const double tol = 1e-12;
    Eigen::VectorXd x_previous = x0;
    options.minimizer_progress_to_stdout = true;
    while(true)
    {
      std::cout << "it: " << it+1 << std::endl;
      Solver::Summary summary_prox;
      Solve(options, &problem, &summary_prox);
      Eigen::VectorXd res(m,1);
      lls.Evaluate(x_optimization,res);
      std::cout << "res: " << res.transpose() << std::endl;
      std::cout << "evol: " << (x_previous-x_optimization).norm() << std::endl;
      it++;
      Eigen::VectorXd grad(n,1);
      lls.Gradient(x_optimization,grad);
      if((x_previous-x_optimization).norm() <= tol && grad.norm() <= tol)
      {
        prox_has_congerged = true; break;
      }
      if(it == max_it)
        break;
      
      x_previous = x_optimization;
      pc_ptr->setReferenceVector(x_optimization);
    }
  }
  
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
