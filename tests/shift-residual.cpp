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
#include "pinocchio/ceres/shift-residual.hpp"

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
    LinearLeastSquareResidual & lls = *lls_ptr;
    Eigen::VectorXd res0(m,1);
    Eigen::MatrixXd jac0(m,n);
    lls.Evaluate(x0,res0,jac0);
    ASSERT_TRUE(res0.isApprox(A*x0-b));
    ASSERT_TRUE(jac0.isApprox(A));
    
    Eigen::VectorXd shift_value(Eigen::VectorXd::Zero(m,1));
    ShiftResidual * shift_ptr = new ShiftResidual(lls_ptr,shift_value);
    
    Eigen::VectorXd x_optimization(x0);
    
    // Build the problem.
    Problem problem;
    problem.AddResidualBlock(shift_ptr, NULL, x_optimization.data());
    
    // Build the problem.
    Problem problem_ref;
    Eigen::VectorXd x_optimization_ref(x0);
    problem_ref.AddResidualBlock(lls_ptr, NULL, x_optimization_ref.data());
    
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
    Solve(options, &problem_ref, &summary);

    ASSERT_TRUE(x_optimization.isApprox(x_optimization_ref));
    
    // Remove affine part in LLS
    lls.setb(Eigen::VectorXd::Zero(m,1));
    shift_ptr->setShiftValue(-b);
    
    Solve(options, &problem, &summary);
    ASSERT_TRUE(x_optimization.isApprox(x_optimization_ref));
  }
  
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


