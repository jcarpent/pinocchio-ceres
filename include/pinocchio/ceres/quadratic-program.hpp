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

#ifndef __pinocchio_ceres_quardratic_program_hpp__
#define __pinocchio_ceres_quardratic_program_hpp__

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "pinocchio/ceres/-constrained-problem.hpp"
#include "pinocchio/ceres/linear-least-square.hpp"

namespace pinocchio
{
  namespace ceres
  {
    ///
    /// \brief Quadratic program of the form \f$ || Ax - b ||_{2}^{2} s.t. Cx = d; Ex \leq f \f$.
    ///
    struct QuadraticProgram : public ConstrainedProblem
    {
      
      typedef Eigen::VectorXd VectorType;
      typedef Eigen::MatrixXd MatrixType;
      typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixType;
      
      template<typename MatrixLike1, typename VectorLike1, typename MatrixLike2, typename VectorLike2>
      QuadraticProgram(const Eigen::MatrixBase<MatrixLike1> & A,
                       const Eigen::MatrixBase<VectorLike1> & b,
                       const Eigen::MatrixBase<MatrixLike> & C,
                       const Eigen::MatrixBase<VectorLike> & d)
      : cost(A,b)
      , equality_constraint(C,d)
      {
        assert(C.rows() == A.rows() && "A and C do not have the right dimensions.");
      }
      
      template<typename MatrixLike>
      void setA(const Eigen::MatrixBase<MatrixLike> & A_new)
      { cost.A = A_new; }
      const MatrixType & getA() const { return cost.A; }
      
      template<typename VectorLike>
      void setb(const Eigen::MatrixBase<VectorLike> & b_new)
      { cost.b = b_new; }
      const VectorType & getb() const { return cost.b; }
      
      template<typename MatrixLike>
      void setC(const Eigen::MatrixBase<MatrixLike> & C_new)
      { equality_constraint.A = C_new; }
      const MatrixType & getC() const { return equality_constraint.A; }
      
      template<typename VectorLike>
      void setd(const Eigen::MatrixBase<VectorLike> & d_new)
      { equality_constraint.b = d_new; }
      const VectorType & getd() const { return equality_constraint.b; }
      
      virtual ::ceres::CostFunction * f()
      {
        return &cost;
      }
      
      virtual ::ceres::CostFunction * g()
      {
        return &equality_constraint;
      }
      
      virtual ::ceres::CostFunction * h()
      {
        return NULL;
      }
      
    protected:
      // data
      
      LinearLeastSquareResidual cost, equality_constraint;
      
    }; // QuadraticProgram
  }
}


#endif // ifndef __pinocchio_ceres_quardratic_program_hpp__
