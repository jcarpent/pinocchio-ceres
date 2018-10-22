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

#ifndef __pinocchio_ceres_proximal_cost_hpp__
#define __pinocchio_ceres_proximal_cost_hpp__

#include "ceres/ceres.h"

namespace pinocchio
{
  namespace ceres
  {
    ///
    /// \brief Proximal term.
    ///
    struct ProximalCost : ::ceres::CostFunction
    {

      typedef Eigen::VectorXd VectorType;
      typedef Eigen::MatrixXd::IdentityReturnType JacobianType;
      typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixType;
      
      template<typename VectorLike>
      ProximalCost(const Eigen::MatrixBase<VectorLike> & x_ref)
      : x_ref(x_ref)
      , jacobian_residual(Eigen::MatrixXd::Identity(x_ref.size(),x_ref.size()))
      {
        mutable_parameter_block_sizes()->push_back((int)x_ref.size());
        set_num_residuals((int)x_ref.size());
      }
      
      template<typename VectorLike, typename ResidualVector, typename JacobianType>
      bool Evaluate(const Eigen::MatrixBase<VectorLike> & x,
                    const Eigen::MatrixBase<ResidualVector> & residual,
                    const Eigen::MatrixBase<JacobianType> & jacobian) const
      {
        const_cast<JacobianType &>(jacobian.derived()) = jacobian_residual;
        Evaluate(x,residual.derived());
        
        return true;
      }
      
      template<typename VectorLike, typename ResidualVector>
      bool Evaluate(const Eigen::MatrixBase<VectorLike> & x,
                    const Eigen::MatrixBase<ResidualVector> & residual) const
      {
        const_cast<ResidualVector &>(residual.derived()).noalias() = x - x_ref;
        
        return true;
      }
      
      bool Evaluate(double const* const* x,
                    double* residuals,
                    double** jacobians) const
      {
        assert(x != NULL && "x is NULL");
        
        const Eigen::Map<const VectorType> x_map(x[0],x_ref.size(),1);
        
        if(jacobians)
        {
          Eigen::Map<VectorType> residuals_map(residuals,x_ref.size(),1);
          Eigen::Map<RowMatrixType> jacobian_map(jacobians[0],x_ref.size(),x_ref.size());
          Evaluate(x_map,residuals_map,jacobian_map);
          
        }
        else
        {
          Eigen::Map<VectorType> residuals_map(residuals,x_ref.size(),1);
          Evaluate(x_map,residuals_map);
        }
        
        return true;
      }
      
      template<typename VectorLike>
      void setReferenceVector(const Eigen::MatrixBase<VectorLike> & _x_ref)
      {
        assert(_x_ref.size() == x_ref.size() && "x_ref does not have the right dimension.");
        x_ref = _x_ref;
      }
      
      const VectorType & getReferenceVector() const
      {
        return x_ref;
      }
      
    protected:
      // data
      
      mutable VectorType res;
      mutable VectorType x_ref;
      mutable JacobianType jacobian_residual;
      
    }; // ProximalCost
  }
}


#endif // ifndef __pinocchio_ceres_proximal_cost_hpp__
