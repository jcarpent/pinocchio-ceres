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

#ifndef __pinocchio_ceres_augmented_lagrangian_problem_hpp__
#define __pinocchio_ceres_augmented_lagrangian_problem_hpp__

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/ceres/constraint-types.hpp"
#include "pinocchio/ceres/shift-residual.hpp"
#include "pinocchio/ceres/eigen.hpp"

namespace pinocchio
{
  namespace ceres
  {
    ///
    /// \brief Base class for equality constrained problem
    ///        of the form \f$ min ||f(x)||_2^{2} s.t. g(x) = 0\f$.
    ///
    struct AugmentedLagrangianProblem
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      
      typedef ::ceres::Solver::Options InnerSolverOptions;
      typedef ::ceres::Solver::Summary InternalSolverSummary;
      typedef ::ceres::Problem InternalSolverProblem;
      typedef std::vector<double*> VectorBlockVariable;
      
      typedef Eigen::VectorXd VectorType;
      
      typedef std::vector<double*> ParameterBlocksType;
      typedef pinocchio::container::aligned_vector<ShiftResidual> VectorShiftResidual;
      
      ///
      /// \brief Structure containing the optimal solution of the Augmented Lagrangian solver.
      ///
      struct Solution
      {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        /// \brief Primal solution.
        VectorType x;
        /// \brief Lagrange multipliers associated to the equality constraints.
        VectorType equality_multipliers;
        /// \brief Lagrange multipliers associated to the inequality constraints.
        VectorType inequality_multipliers;
        
        /// \brief Penalty on the equality constraints at the end of the resolution.
        double mu_equality;
        /// \brief Penalty on the inequality constraints at the end of the resolution.
        double mu_inequality;
        
        /// \brief Number of outer iterations.
        int num_outer_iterations;
        
        /// \brief Number of inner iterations.
        int num_inner_iterations;
        
        /// \brief Optimization status.
        ::ceres::TerminationType status;
      };
      
      ///
      /// \brief Options for the Augmented Lagrangian solver.
      ///
      struct Options
      {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        /// \brief Options for the internal solver
        InnerSolverOptions internal_options;
        
        /// \brief Initial penalty on the equality constraints.
        double mu0_equality = 10.;
        
        /// \brief Initial penalty on the inequality constraints.
        double mu0_inequality = 10.;
        
        /// \brief Penalty factor update.
        double penalty_factor_update = 10.;
        
        /// \brief Maximal number of iterations.
        int max_iterations = 100;
        
        /// \brief Initial tolerance on the inner iterations.
        double omega0 = 1./std::max(mu0_equality,mu0_inequality);
        
        /// \brief Initial tolerance on the equality constraints.
        double eta0_equality = 1./std::pow(mu0_equality,0.1);
        
        /// \brief Initial tolerance on the inequality constraints.
        double eta0_inequality = 1./std::pow(mu0_inequality,0.1);
        
        /// \brief Check gradients during the inner iteration.
        bool check_gradients = false;
        
        /// \brief Tolerance on the gradient of the Auglab Problem at optimum.
        double omega_opt = 1e-6;
        
        /// \brief Tolerance on the equality constraints at optimum.
        double eta_opt_equality = 1e-6;
        
        /// \brief Tolerance on the inequality constraints at optimum.
        double eta_opt_inequality = 1e-6;
        
        /// \brief Report progress on std::cout.
        bool outer_loop_progress_to_stdout = false;
        
      };
      
      /// \brief Default constructor.
      AugmentedLagrangianProblem();
      
      /// \brief Add a cost residual term.
      void AddCostResidualBlock(::ceres::CostFunction* cost_function,
                                ::ceres::LossFunction* loss_function,
                                const ParameterBlocksType & parameter_blocks);
      
      /// \brief Convenience method for adding a cost residual with a single parameter block.
      void AddCostResidualBlock(::ceres::CostFunction* cost_function,
                                ::ceres::LossFunction* loss_function,
                                double* x0);
      
      /// \brief Convenience method for adding a cost residual with two parameter blocks.
      void AddCostResidualBlock(::ceres::CostFunction* cost_function,
                                ::ceres::LossFunction* loss_function,
                                double* x0, double* x1);
      
      /// \brief Convenience method for adding a cost residual with three parameter blocks.
      void AddCostResidualBlock(::ceres::CostFunction* cost_function,
                                ::ceres::LossFunction* loss_function,
                                double* x0, double* x1, double * x2);
      
      /// \brief Add an equality constrained residual block.
      void AddEqualityConstraintResidualBlock(::ceres::CostFunction* residual,
                                              const ParameterBlocksType & parameter_blocks);
      
      /// \brief Convenience method for adding an equality constrained residual block with a single parameter block.
      void AddEqualityConstraintResidualBlock(::ceres::CostFunction* residual,
                                              double* x0);
      
      /// \brief Convenience method for adding an equality constrained residual block with a single parameter block.
      void AddEqualityConstraintResidualBlock(::ceres::CostFunction* residual,
                                              double* x0, double* x1);
      
      /// \brief Convenience method for adding an equality constrained residual block with a single parameter block.
      void AddEqualityConstraintResidualBlock(::ceres::CostFunction* residual,
                                              double* x0, double* x1, double * x2);
      
      ///
      /// \brief Solve the problem from an initial solution.
      ///
      /// \returns the optimal solution.
      ///
      Solution Solve(const Options & options);
      
      /// \brief Set the local parametrization for one of the parameter blocks
      void SetParameterization(double *values, ::ceres::LocalParameterization *local_parameterization)
      {
        auglag_ceres_problem->SetParameterization(values,local_parameterization);
      }
      
      /// \brief Working variables for the AugLag problem.
      struct WorkingSet
      {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        /// \brief Init from options.
        WorkingSet(const Options & options);
        
        /// \brief Current iteration.
        int it;
        
        /// \brief Primal variable.
        Eigen::VectorXd x_current;
        
        /// \brief Equality multipliers.
        Eigen::VectorXd equality_multipliers;
        
        /// \brief Inequality multipliers.
        Eigen::VectorXd inequality_multipliers;
        
        /// \brief Equality penalty value.
        double mu_equality;
        
        /// \brief Inquality penalty value.
        double mu_inequality;
        
        /// \brief Initial tolerance on the inner iterations.
        double omega;
        
        /// \brief Tolerance on the equality constraints.
        double eta_equality;
        
        /// \brief Tolerance on the inequality constraints.
        double eta_inequality;
      };
      
      int NumParameters() const
      {
        return auglag_ceres_problem->NumParameters();
      }
      
      int NumEffectiveParameters() const
      {
        return auglag_ceres_problem->NumEffectiveParameters();
      }
      
      int NumEqualityBlocks() const
      {
        return (int)shifted_equality_constraints_vec.size();
      }
      
      int NumEqualities() const
      {
        return num_equalities;
      }
      
      int NumInequalityBlocks() const
      {
        return (int)shifted_inequality_constraints_vec.size();
      }
      
      int NumInequalities() const
      {
        return num_inequalities;
      }
      
      ///
      /// \brief Evaluate the cost and the gradient at a given entry point x.
      ///
      /// \param[out] cost The return cost value of the Augmented Lagrangian.
      /// \param[out] gradient The value of the gradient of the Augmented Lagrangian.
      ///
      void Evaluate(double & cost,
                    Eigen::VectorXd & gradient);
      
      ///
      /// \brief Evaluate the gradient only of the cost function.
      ///
      /// \param[out] cost The return cost value.
      /// \param[out] gradient The value of the gradient of the cost function.
      ///
      void EvaluateCostGradient(double & cost,
                                Eigen::VectorXd & cost_gradient);
      
      ///
      /// \brief Evaluate the equality constraints results (gradient, residuals, Jacobian).
      ///
      /// \param[out] constraint_residuals_ptr Pointer on the vector containing the constraint residuals.
      /// \param[out] constraint_gradient_ptr Pointer on the gradient vector of the equality constraints.
      /// \param[out] constraint_jacobian_ptr Pointer on the Jacobian matrix of the equality constraints.
      ///
      void EvaluateEqualityConstraints(std::vector<double> * constraint_residuals_ptr,
                                       std::vector<double> * constraint_gradient_ptr,
                                       ::ceres::CRSMatrix * constraint_jacobian_ptr);
      
      ///
      /// \brief Evaluate the equality constraints gradient.
      ///
      /// \param[out] constraint_gradient Gradient vector of the equality constraints.
      ///
      void EvaluateEqualityConstraintsGradient(Eigen::VectorXd & constraint_gradient);
      
      ///
      /// \brief Evaluate the equality constraints residual.
      ///
      /// \param[out] constraint_residuals Vector containing the constraint residuals.
      ///
      void EvaluateEqualityConstraintsResiduals(Eigen::VectorXd & constraint_residuals);
      
      ///
      /// \brief Evaluate the Jacobian of the equality constraints
      ///
      /// \param[out] constraint_jacobian Jacobian matrix of the equality constraints.
      ///
      void EvaluateEqualityConstraintsJacobian(::ceres::CRSMatrix & constraint_jacobian);
      
      virtual ~AugmentedLagrangianProblem();
      
    protected:
      
      /// \brief Init the Augmented Lagrangian problem.
      ///
      /// \details Set up the Ceres problem and do the memory allocation
      void initAugLagProblem();

      /// \brief Scaling factor in front of the equality augmented terms.
      ///
      /// \details This factor is common for all the equality constraints.
      ::ceres::LossFunctionWrapper scaling_factor_equality;
      
      /// \brief Scaling factor in front of the inequality augmented terms.
      ///
      /// \details This factor is common for all the inequality constraints.
      ::ceres::LossFunctionWrapper scaling_factor_inequality;
      
      /// \brief The inner loop problem to be solved.
      ::ceres::Problem * auglag_ceres_problem;
      
      /// \brief Vector of shifted equality constraints.
      ///
      /// \details Each equality constraint has its own multiplier vectors, which
      ///          corresponds to the shift operated by the ShiftResidual cost.
      VectorShiftResidual shifted_equality_constraints_vec;
      
      /// \brief Vector of shifted equality constraints.
      VectorShiftResidual shifted_inequality_constraints_vec;
      
      typedef ::ceres::ResidualBlockId ResidualBlockId;
      typedef std::vector<ResidualBlockId> VectorResidualBlockId;
      
      /// \brief Vector of residuals for the cost terms.
      VectorResidualBlockId cost_residual_blocks;
      
      /// \brief Vector of residuals for the equality constraints.
      VectorResidualBlockId equality_constraint_residual_blocks;
      
      /// \brief Vector of residuals for the inequality constraints.
      VectorResidualBlockId inequality_constraint_residual_blocks;
      
      /// \brief Vector of residuals for the active inequality constraints.
      VectorResidualBlockId active_inequality_constraint_residual_blocks;
     
      /// \brief Equality constraints segment info.
      /// \details Provides the segment position inside residual vector or multipliers.
      std::vector<EigenSegmentInfo> equality_segment_info;
      
    private:
      
      /// \brief Update the penalty value for each ScalingLoss.
      void updatePenalty(::ceres::LossFunctionWrapper & scaling_factor,
                         const double & penalty);
      
      /// \brief Update the multipliers values of the shifted residual.
      void setEqualityMultipliers(const Eigen::VectorXd & multipliers,
                                  const double & penalty);
      
      /// \brief Number of equalities.
      int num_equalities;
      
      /// \brief Number of inequalities.
      int num_inequalities;
      
    }; // AugmentedLagrangianProblem
  }
}

#endif // ifndef __pinocchio_ceres_augmented_lagrangian_problem_hpp__
