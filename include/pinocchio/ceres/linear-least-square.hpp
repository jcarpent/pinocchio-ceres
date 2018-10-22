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

#ifndef __pinocchio_ceres_linear_least_square_hpp__
#define __pinocchio_ceres_linear_least_square_hpp__

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace pinocchio
{
  namespace ceres
  {
    ///
    /// \brief Proximal term.
    ///
    struct LinearLeastSquareResidual : ::ceres::CostFunction
    {
      
      typedef Eigen::VectorXd VectorType;
      typedef Eigen::MatrixXd MatrixType;
      typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixType;

      template<typename MatrixLike, typename VectorLike>
      LinearLeastSquareResidual(const Eigen::MatrixBase<MatrixLike> & A,
                                const Eigen::MatrixBase<VectorLike> & b)
      : A(A)
      , b(b)
      {
        assert(A.rows() == b.size() && "A and b do not have the correct dimension");
        mutable_parameter_block_sizes()->push_back((int)A.cols());
        set_num_residuals((int)A.rows());
      }
      
      template<typename VectorLike, typename ResidualVectorLike, typename JacobianLike>
      bool Evaluate(const Eigen::MatrixBase<VectorLike> & x,
                    const Eigen::MatrixBase<ResidualVectorLike> & residual,
                    const Eigen::MatrixBase<JacobianLike> & jacobian) const
      {
        const_cast<JacobianLike &>(jacobian.derived()) = A;
        Evaluate(x,residual.derived());
        
        return true;
      }
      
      template<typename VectorLike, typename ResidualVector>
      bool Evaluate(const Eigen::MatrixBase<VectorLike> & x,
                    const Eigen::MatrixBase<ResidualVector> & residual) const
      {
        const_cast<ResidualVector &>(residual.derived()).noalias() = A*x - b;
        
        return true;
      }
      
      bool Evaluate(double const* const* x,
                    double* residuals,
                    double** jacobians) const
      {
        assert(x != NULL && "x is NULL");
        
        const Eigen::Map<const VectorType> x_map(x[0],A.cols(),1);
        
        if(jacobians)
        {
          Eigen::Map<RowMatrixType> jacobian_map(jacobians[0],A.rows(),A.cols());
          Eigen::Map<VectorType> residuals_map(residuals,A.rows(),1);
          Evaluate(x_map,residuals_map,jacobian_map);
        }
        else
        {
          Eigen::Map<VectorType> residuals_map(residuals,A.rows(),1);
          Evaluate(x_map,residuals_map);
        }
        
        return true;
      }
      
      template<typename VectorLike, typename GradientVector>
      bool Gradient(const Eigen::MatrixBase<VectorLike> & x,
                    const Eigen::MatrixBase<GradientVector> & gradient) const
      {
        const_cast<GradientVector &>(gradient.derived()).noalias() = A.transpose() * (A*x - b);
        
        return true;
      }
      
      template<typename MatrixLike>
      void setA(const Eigen::MatrixBase<MatrixLike> A_new) { A = A_new; }
      const MatrixType & getA() const { return A; }
      
      template<typename VectorLike>
      void setb(const Eigen::MatrixBase<VectorLike> b_new) { b = b_new; }
      const VectorType & getb() const { return b; }
      
    protected:
      // data

      MatrixType A; VectorType b;
      
    }; // LinearLeastSquareResidual
  }
}


#endif // ifndef __pinocchio_ceres_linear_least_square_hpp__
