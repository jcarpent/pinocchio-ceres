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

#ifndef __pinocchio_ceres_proximal_configuration_cost_hpp__
#define __pinocchio_ceres_proximal_configuration_cost_hpp__

#include "ceres/ceres.h"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <pinocchio/algorithm/joint-configuration.hpp>

namespace pinocchio
{
  namespace ceres
  {
    ///
    /// \brief Proximal term on the configuration of the robot.
    ///        This class allows to penalize the distance of the current configuration
    ///        to a reference one and called q_ref.
    ///
    template<typename _Scalar, int _Options>
    struct ProximalConfigurationCost : ::ceres::CostFunction
    {
      typedef _Scalar Scalar;
      enum { Options = _Options };
      
      typedef se3::ModelTpl<Scalar,Options> Model;
      typedef se3::DataTpl<Scalar,Options> Data;
      
      typedef typename Model::ConfigVectorType ConfigVectorType;
      typedef typename Model::TangentVectorType TangentVectorType;
      typedef TangentVectorType ResidualVectorType;
      typedef typename Data::MatrixXs JacobianType;
      
      template<typename ConfigVectorLike>
      ProximalConfigurationCost(const Model & model,
                                const Eigen::MatrixBase<ConfigVectorLike> & q_ref)
      : model(model)
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
        Evaluate(q,residual.derived());
        
        return true;
      }
      
      template<typename ConfigVectorType, typename ResidualVector>
      bool Evaluate(const Eigen::MatrixBase<ConfigVectorType> & q,
                    const Eigen::MatrixBase<ResidualVector> & residual) const
      {
        // just call rnea
        EIGEN_CONST_CAST(ResidualVector,residual) = se3::difference(model,q_ref,q);
        
        return true;
      }
      
      bool Evaluate(double const* const* x,
                    double* residuals,
                    double** jacobians) const
      {
        assert(x != NULL && "x is NULL");
        
        const Eigen::Map<const ConfigVectorType> q_map(x[0],model.nq,1);
        
        if(jacobians)
        {
          Evaluate(q_map,res,jacobian_residual);
          Eigen::Map<ResidualVectorType> residuals_map(residuals,model.nv,1);
          residuals_map = res;
          
          Eigen::Map<typename EIGEN_PLAIN_ROW_MAJOR_TYPE(JacobianType)>(jacobians[0],model.nv,model.nq).leftCols(model.nv)
          = jacobian_residual;
        }
        else
        {
          Evaluate(q_map,res);
          Eigen::Map<ResidualVectorType> residuals_map(residuals,model.nv,1);
          residuals_map = res;
        }
        
        return true;
      }
      
      template<typename ConfigVectorLike>
      void setReferenceConfiguration(const Eigen::MatrixBase<ConfigVectorLike> & _q_ref)
      {
        assert(_q_ref.size() == model.nq && "q_ref does not have the right dimension.");
        q_ref = _q_ref;
      }
      
      const ConfigVectorType & getReferenceConfiguration() const
      {
        return q_ref;
      }
      
    protected:
      // data
      const Model & model;
      
      mutable ResidualVectorType res;
      mutable ConfigVectorType q_ref;
      mutable JacobianType jacobian_residual;
      
    }; // ProximalConfigurationCost
  }
}


#endif // ifndef __pinocchio_ceres_proximal_configuration_cost_hpp__
