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

#ifndef __pinocchio_ceres_local_parameterization_hpp__
#define __pinocchio_ceres_local_parameterization_hpp__

#include <Eigen/Core>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <ceres/local_parameterization.h>

namespace pinocchio
{
  namespace ceres
  {
    
    template<typename _Scalar, int _Options>
    class ModelLocalParameterization : public ::ceres::LocalParameterization
    {
    public:
      typedef _Scalar Scalar;
      enum { Options = _Options };
      typedef se3::ModelTpl<Scalar,Options> Model;
      typedef se3::DataTpl<Scalar,Options> Data;
      typedef typename Model::ConfigVectorType ConfigVectorType;
      typedef typename Model::TangentVectorType TangentVectorType;
      
      typedef typename Data::MatrixXs MatrixXs;
      typedef typename EIGEN_PLAIN_ROW_MAJOR_TYPE(MatrixXs) RowMatrixXs;
      
      ModelLocalParameterization(const Model & model)
      : m_model(model)
      {}
      
      int GlobalSize() const { return m_model.nq; }
      int LocalSize() const { return m_model.nv; }
      
      const Model & model() const { return m_model; }
      
      bool Plus(const double * x,
                const double * delta,
                double * x_plus_delta) const
      {
        Eigen::Map<const ConfigVectorType> q(x,m_model.nq);
        Eigen::Map<const TangentVectorType> delta_q(delta,m_model.nv);
        
        Eigen::Map<ConfigVectorType> q_plus(x_plus_delta,m_model.nq);
        q_plus = se3::integrate(m_model,q,delta_q);
        return true;
      }
      
      bool ComputeJacobian(const double * x, double * jacobian) const
      {
        Eigen::Map<const ConfigVectorType> q(x,m_model.nq);
        Eigen::Map<RowMatrixXs> jac(jacobian,m_model.nq,m_model.nv);
        se3::integrateCoeffWiseJacobian(m_model,q,jac);
        return true;
      }
      
      ~ModelLocalParameterization() {}
      
    protected:
      
      const Model & m_model;
    };
    
    
  } // namespace ceres
} // namespace pinocchio

#endif // __pinocchio_ceres_local_parameterization_hpp__
