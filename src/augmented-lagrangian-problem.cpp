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

#include "pinocchio/ceres/augmented-lagrangian-problem.hpp"

namespace pinocchio
{
  namespace ceres
  {
    
    AugmentedLagrangianProblem::WorkingSet::WorkingSet(const Options & options)
    : it(0)
    , mu_equality(options.mu0_equality)
    , mu_inequality(options.mu0_inequality)
    , omega(options.omega0)
    , eta_equality(options.eta0_equality)
    , eta_inequality(options.eta0_inequality)
    {}
    
    AugmentedLagrangianProblem::AugmentedLagrangianProblem()
    : scaling_factor_equality(new ::ceres::ScaledLoss(NULL,NAN,::ceres::DO_NOT_TAKE_OWNERSHIP),
                              ::ceres::TAKE_OWNERSHIP)
    , scaling_factor_inequality(new ::ceres::ScaledLoss(NULL,NAN,::ceres::DO_NOT_TAKE_OWNERSHIP),
                                ::ceres::TAKE_OWNERSHIP)
    , num_equalities(0), num_inequalities(0)
    {
      // Init AugLag problem
      initAugLagProblem();
    }
    
    void AugmentedLagrangianProblem::initAugLagProblem()
    {
      ::ceres::Problem::Options options;
      options.cost_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
      options.loss_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
      options.loss_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
      auglag_ceres_problem = new ::ceres::Problem(options);
    }
    
    void AugmentedLagrangianProblem::AddCostResidualBlock(::ceres::CostFunction* cost_function,
                                                          ::ceres::LossFunction* loss_function,
                                                          const ParameterBlocksType & parameter_blocks)
    {
      ResidualBlockId residual_block_id = auglag_ceres_problem->AddResidualBlock(cost_function,loss_function,parameter_blocks);
      
      cost_residual_blocks.push_back(residual_block_id);
    }
    
    void AugmentedLagrangianProblem::AddCostResidualBlock(::ceres::CostFunction* cost_function,
                                                          ::ceres::LossFunction* loss_function,
                                                          double* x0)
    {
      ParameterBlocksType parameter_blocks; parameter_blocks.push_back(x0);
      AddCostResidualBlock(cost_function,loss_function,parameter_blocks);
    }
    
    void AugmentedLagrangianProblem::AddCostResidualBlock(::ceres::CostFunction* cost_function,
                                                          ::ceres::LossFunction* loss_function,
                                                          double* x0, double* x1)
    {
      ParameterBlocksType parameter_blocks;
      parameter_blocks.push_back(x0); parameter_blocks.push_back(x1);
      AddCostResidualBlock(cost_function,loss_function,parameter_blocks);
    }
    
    void AugmentedLagrangianProblem::AddCostResidualBlock(::ceres::CostFunction* cost_function,
                                                          ::ceres::LossFunction* loss_function,
                                                          double* x0, double* x1, double* x2)
    {
      ParameterBlocksType parameter_blocks;
      parameter_blocks.push_back(x0); parameter_blocks.push_back(x1); parameter_blocks.push_back(x2);
      AddCostResidualBlock(cost_function,loss_function,parameter_blocks);
    }
    
    void AugmentedLagrangianProblem::
    AddEqualityConstraintResidualBlock(::ceres::CostFunction * residual,
                                       const ParameterBlocksType & parameter_blocks)
    {
      ShiftResidual shifted_residual(residual);

      shifted_equality_constraints_vec.push_back(shifted_residual);
      ResidualBlockId residual_block_id =
      auglag_ceres_problem->AddResidualBlock(&shifted_equality_constraints_vec.back(),
                                             &scaling_factor_equality,
                                             parameter_blocks);
      
      const int num_residuals = residual->num_residuals();
      num_equalities += num_residuals;
      
      equality_constraint_residual_blocks.push_back(residual_block_id);
      
      EigenSegmentInfo segment_info;
      segment_info.start = equality_segment_info.size() == 0
      ? 0 : equality_segment_info.back().start + equality_segment_info.back().length;
      segment_info.length = num_residuals;
      
      equality_segment_info.push_back(segment_info);
    }
    
    void AugmentedLagrangianProblem::
    AddEqualityConstraintResidualBlock(::ceres::CostFunction* residual,
                                       double* x0)
    {
      ParameterBlocksType parameter_blocks;
      parameter_blocks.push_back(x0);
      AddEqualityConstraintResidualBlock(residual,parameter_blocks);
    }
    
    void AugmentedLagrangianProblem::
    AddEqualityConstraintResidualBlock(::ceres::CostFunction* residual,
                                       double* x0, double* x1)
    {
      ParameterBlocksType parameter_blocks;
      parameter_blocks.push_back(x0); parameter_blocks.push_back(x1);
      AddEqualityConstraintResidualBlock(residual,parameter_blocks);
    }
    
    void AugmentedLagrangianProblem::
    AddEqualityConstraintResidualBlock(::ceres::CostFunction* residual,
                                       double* x0, double* x1, double * x2)
    {
      ParameterBlocksType parameter_blocks;
      parameter_blocks.push_back(x0); parameter_blocks.push_back(x1); parameter_blocks.push_back(x2);
      AddEqualityConstraintResidualBlock(residual,parameter_blocks);
    }
    
    AugmentedLagrangianProblem::Solution
    AugmentedLagrangianProblem::Solve(const Options & options)
    {
      // Initilize local variables
      const bool with_equalities = NumEqualities() > 0;
      const bool with_inequalities = NumInequalities() > 0;
      
      Eigen::VectorXd gradient(NumParameters(),1);
      double auglag_value;
      double max_norm_gradient;
      
      Eigen::VectorXd equality_constraint_residuals_value(NumEqualities(),1);
      double max_norm_equality_constraints = 1e10;
        if(!with_equalities)
        max_norm_equality_constraints = 0;
      
      Eigen::VectorXd inequality_constraint_residuals_value(NumInequalities(),1);
      double max_norm_inequality_constraints = 1e10;
      if(!with_inequalities)
        max_norm_inequality_constraints = 0;
      
      // Initialize working set of variables from options
      WorkingSet working_set(options);
      int current_it = 0;
      Eigen::VectorXd & x_current = working_set.x_current;
      x_current.resize(NumParameters());
      
      Eigen::VectorXd x_candidate(NumParameters(),1);
      
      // Initialize evaluate options for problem
      ::ceres::Problem::EvaluateOptions evaluate_options_inner_loop;
      evaluate_options_inner_loop.apply_loss_function = true;
      evaluate_options_inner_loop.num_threads = 1;
      
      if(with_equalities)
      {
        assert(working_set.mu_equality > 0. && "mu_equality must be positive");
        updatePenalty(scaling_factor_equality,working_set.mu_equality);
        
        working_set.equality_multipliers.resize(NumEqualities());
        working_set.equality_multipliers.setOnes();
        setEqualityMultipliers(working_set.equality_multipliers,working_set.mu_equality);
      }
      
      if(with_inequalities)
      {
        assert(working_set.mu_inequality > 0. && "mu_inequality must be positive");
        updatePenalty(scaling_factor_inequality,working_set.mu_inequality);
        
        assert(false && "Not yet implemented");
      }
      
      // Inner loop solver options
      ::ceres::Solver::Options inner_solver_options = options.internal_options;
      
      ::ceres::Solver::Summary inner_solver_summary; // will contain intermediate result
      
      bool tolerance_on_gradient_norm_fullfiled_at_optimum = false;
     
      bool tolerance_on_equality_constraints_fullfiled_at_optimum = false;
      if(!with_equalities)
        tolerance_on_equality_constraints_fullfiled_at_optimum = true;
      
      bool tolerance_on_inequality_constraints_fullfiled_at_optimum = false;
      if(!with_inequalities)
        tolerance_on_inequality_constraints_fullfiled_at_optimum = true;
      
      if(with_equalities)
      {
        EvaluateEqualityConstraintsResiduals(equality_constraint_residuals_value);
        if(options.outer_loop_progress_to_stdout)
          std::cout << "initial equality values:\n" << equality_constraint_residuals_value.transpose() << std::endl;
      }
      
      // Main loop
      while(true)
      {
        bool tolerance_on_equality_constraints_fullfiled = false;
        if(!with_equalities)
          tolerance_on_equality_constraints_fullfiled = true;
        
        bool tolerance_on_inequality_constraints_fullfiled = false;
        if(!with_inequalities)
          tolerance_on_inequality_constraints_fullfiled = true;
        
        current_it += 1;
        if(options.outer_loop_progress_to_stdout)
          std::cout << "It - " << current_it << std::endl;
        working_set.it = current_it;
        
        // Solve inner loop
        ::ceres::Solve(inner_solver_options,
                       auglag_ceres_problem,
                       &inner_solver_summary);
        
        if(options.outer_loop_progress_to_stdout)
          std::cout << inner_solver_summary.FullReport() << std::endl;
        
        // Evaluate the cost and the gradient
        Evaluate(auglag_value,gradient);
        if(options.outer_loop_progress_to_stdout) {
          std::cout << "auglag_value:" << auglag_value << std::endl;
          std::cout << "gradient:" << gradient.transpose() << std::endl;
        }
        
        // Compute the maximal norm of the gradient
        max_norm_gradient = gradient.lpNorm<Eigen::Infinity>();
        if(options.outer_loop_progress_to_stdout)
        std::cout << "norm gradient: " << max_norm_gradient << std::endl;
        if(not (max_norm_gradient <= working_set.omega))
        {
          if(options.outer_loop_progress_to_stdout)
            std::cerr << "The norm of the gradient is too big" << std::endl;
        }
        
        if(max_norm_gradient <= options.omega_opt)
          tolerance_on_gradient_norm_fullfiled_at_optimum = true;
        
        // Evaluate constraints value
        if(with_equalities)
        {
        EvaluateEqualityConstraintsResiduals(equality_constraint_residuals_value);
          max_norm_equality_constraints = equality_constraint_residuals_value.lpNorm<Eigen::Infinity>();
          if(options.outer_loop_progress_to_stdout) {
            std::cout << "equality constraint norm: " << max_norm_equality_constraints << std::endl;
            std::cout << "equality constraint value: " << equality_constraint_residuals_value.transpose() << std::endl;
          }
        }
        
        // Else update multipliers
        if(with_equalities)
        {
          if(max_norm_equality_constraints
             <= std::max(working_set.eta_equality,options.eta_opt_equality))
          {
            tolerance_on_equality_constraints_fullfiled = true;
            working_set.eta_equality /= (1 + std::pow(working_set.mu_equality,0.9));
            
            // Update equality multipliers
            if(options.outer_loop_progress_to_stdout)
              std::cout << "Update equality multipliers" << std::endl;
            working_set.equality_multipliers.noalias()
            += working_set.mu_equality * equality_constraint_residuals_value;
            setEqualityMultipliers(working_set.equality_multipliers,working_set.mu_equality);
            if(options.outer_loop_progress_to_stdout)
              std::cout << "equality multipliers:\n" << working_set.equality_multipliers.transpose() << std::endl;
            
            if(max_norm_equality_constraints <= options.eta_opt_equality)
              tolerance_on_equality_constraints_fullfiled_at_optimum = true;
          }
          else
          {
            // Increase penalty factor
            if(options.outer_loop_progress_to_stdout)
              std::cout << "Increase penalty factor" << std::endl;
            working_set.mu_equality *= options.penalty_factor_update;
            updatePenalty(scaling_factor_equality, working_set.mu_equality);
            
            setEqualityMultipliers(working_set.equality_multipliers,working_set.mu_equality);
            if(options.outer_loop_progress_to_stdout)
              std::cout << "mu: " << working_set.mu_equality << std::endl;
            
            // May increase or decrease expected constraint contraction
            working_set.eta_equality = options.eta0_equality / (1 + std::pow(working_set.mu_equality,0.1));
          }
            
        }
        
        if(tolerance_on_gradient_norm_fullfiled_at_optimum
           && tolerance_on_equality_constraints_fullfiled_at_optimum
           && tolerance_on_inequality_constraints_fullfiled_at_optimum)
        {
          if(options.outer_loop_progress_to_stdout)
            std::cout << "All the tolerance for optimality have been obtained" << std::endl;
          break;
        }
        
    
        if(current_it == options.max_iterations)
        {
          if(options.outer_loop_progress_to_stdout)
            std::cout << "Maximum number of iterations reached!" << std::endl;
          break;
        }
      }
      
      // Fill and return solution
      Solution sol;
      sol.num_outer_iterations = working_set.it;
      sol.equality_multipliers = working_set.equality_multipliers;
      sol.inequality_multipliers = working_set.inequality_multipliers;
      
      return sol;
    }
    
    void AugmentedLagrangianProblem::updatePenalty(::ceres::LossFunctionWrapper & scaling_factor,
                                                   const double & penalty)
    {
      scaling_factor.Reset(new ::ceres::ScaledLoss(NULL,
                                                   penalty,
                                                   ::ceres::DO_NOT_TAKE_OWNERSHIP),
                           ::ceres::TAKE_OWNERSHIP);
    }
    
    void AugmentedLagrangianProblem::setEqualityMultipliers(const Eigen::VectorXd & multipliers,
                                                            const double & penalty)
    {
      assert(multipliers.size() == NumEqualities()
             && "The vector of multipliers does not have the right size");
      assert(penalty >= 0. && "The penalty value is non positive");
      
      for(size_t k = 0; k < shifted_equality_constraints_vec.size(); ++k)
      {
        ShiftResidual & shift_residual = shifted_equality_constraints_vec[k];
        const EigenSegmentInfo & segment_info = equality_segment_info[k];
        shift_residual.setShiftValue(multipliers.segment(segment_info.start,segment_info.length)/penalty);
      }
    }
    
    void AugmentedLagrangianProblem::Evaluate(double & cost,
                                              Eigen::VectorXd & auglag_gradient)
    {
      const Eigen::DenseIndex dim = NumParameters();
      std::vector<double> gradient_vector((size_t)dim,0.);
      auglag_ceres_problem->Evaluate(::ceres::Problem::EvaluateOptions(),
                                    &cost,NULL,&gradient_vector,NULL);
      
      // copy back the value of gradient
      auglag_gradient = Eigen::Map<Eigen::VectorXd>(gradient_vector.data(),dim,1);
    }
    
    void AugmentedLagrangianProblem::EvaluateCostGradient(double & cost,
                                                          Eigen::VectorXd & cost_gradient)
    {
      const bool apply_loss_function = true;
      ::ceres::Problem::EvaluateOptions evaluate_options;
      evaluate_options.apply_loss_function = apply_loss_function;
      evaluate_options.residual_blocks = cost_residual_blocks;
      
      std::vector<double> std_cost_gradient((size_t)NumParameters(),0.);
      
      auglag_ceres_problem->Evaluate(evaluate_options,
                                     &cost, NULL,
                                     &std_cost_gradient, NULL);
      
      cost_gradient = Eigen::Map<Eigen::VectorXd>(std_cost_gradient.data(),
                                                  (Eigen::DenseIndex)NumParameters(),1);
    }
    
    void AugmentedLagrangianProblem::
    EvaluateEqualityConstraints(std::vector<double> * constraint_residuals_ptr,
                                std::vector<double> * constraint_gradient_ptr,
                                ::ceres::CRSMatrix * constraint_jacobian_ptr)
    {
      if(NumEqualities() == 0)
      {
        constraint_residuals_ptr->clear();
        constraint_gradient_ptr->clear();
        constraint_jacobian_ptr->num_rows = 0;
        constraint_jacobian_ptr->num_cols = NumEffectiveParameters();
        return;
      }
      
      const bool apply_loss_function = false;
      ::ceres::Problem::EvaluateOptions evaluate_options;
      evaluate_options.apply_loss_function = apply_loss_function;
      evaluate_options.residual_blocks = equality_constraint_residual_blocks;
      
      for(VectorShiftResidual::iterator it = shifted_equality_constraints_vec.begin();
          it != shifted_equality_constraints_vec.end(); ++it)
      {
        it->applyShift(false);
      }
      
      auglag_ceres_problem->Evaluate(evaluate_options,
                                     NULL, constraint_residuals_ptr,
                                     constraint_gradient_ptr, constraint_jacobian_ptr);
      
      for(VectorShiftResidual::iterator it = shifted_equality_constraints_vec.begin();
          it != shifted_equality_constraints_vec.end(); ++it)
      {
        it->applyShift(true);
      }
    }
    
    void AugmentedLagrangianProblem::
    EvaluateEqualityConstraintsGradient(Eigen::VectorXd & constraint_gradient)
    {
      std::vector<double> std_constraint_gradient((size_t)NumEffectiveParameters(),0.);
  
      EvaluateEqualityConstraints(NULL,&std_constraint_gradient,NULL);
      
      constraint_gradient = Eigen::Map<Eigen::VectorXd>(std_constraint_gradient.data(),
                                                        (Eigen::DenseIndex)std_constraint_gradient.size(),1);
      
    }
    
    void AugmentedLagrangianProblem::
    EvaluateEqualityConstraintsResiduals(Eigen::VectorXd & constraint_residuals)
    {
      
      std::vector<double> std_constraint_residuals((size_t)NumEqualities(),0.);
      
      EvaluateEqualityConstraints(&std_constraint_residuals,NULL,NULL);
      
      constraint_residuals = Eigen::Map<Eigen::VectorXd>(std_constraint_residuals.data(),
                                                         (Eigen::DenseIndex)std_constraint_residuals.size(),1);
      
    }
    
    void AugmentedLagrangianProblem::
    EvaluateEqualityConstraintsJacobian(::ceres::CRSMatrix & constraint_jacobian)
    {
      
      EvaluateEqualityConstraints(NULL,NULL,&constraint_jacobian);
      
    }
    
    AugmentedLagrangianProblem::~AugmentedLagrangianProblem()
    {
      delete auglag_ceres_problem;
    }
    
  }
}

