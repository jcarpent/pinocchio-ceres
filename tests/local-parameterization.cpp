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

#include "pinocchio/ceres/local-parameterization.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"

namespace
{
  
  TEST(LocalParameterization,HumanoidSimple)
  {
    using namespace pinocchio::ceres;
    typedef double Scalar;
    enum { Options = 0 };
    
    typedef ModelLocalParameterization<Scalar,Options> LocalParameterization;
    typedef LocalParameterization::Model Model;
    typedef LocalParameterization::Data Data;
    
    Model model;
    se3::buildModels::humanoidSimple(model);
    
    LocalParameterization local_para(model);
    ASSERT_EQ(local_para.GlobalSize(),model.nq);
    ASSERT_EQ(local_para.LocalSize(),model.nv);
    
    typedef Model::ConfigVectorType ConfigVectorType;
    typedef Model::TangentVectorType TangentVectorType;
    
    model.lowerPositionLimit.head<7>().fill(-1.);
    model.upperPositionLimit.head<7>().fill(1.);
    ConfigVectorType q = se3::randomConfiguration(model);
    TangentVectorType v = TangentVectorType::Random(model.nv,1);
    
    ConfigVectorType q_plus_ref = se3::integrate(model,q,v);
    ConfigVectorType q_plus(model.nq,1);
    local_para.Plus(q.data(),v.data(),q_plus.data());
    ASSERT_TRUE(q_plus.isApprox(q_plus_ref));
    
    typedef LocalParameterization::MatrixXs MatrixXs;
    typedef LocalParameterization::RowMatrixXs RowMatrixXs;
    
    RowMatrixXs jac_row_major(model.nq,model.nv); jac_row_major.setZero();
    local_para.ComputeJacobian(q.data(),jac_row_major.data());

    MatrixXs jac_fd(model.nq,model.nv);
    
    const Scalar eps = 1e-8;
    TangentVectorType v_eps = TangentVectorType::Zero(model.nv,1);
    for(int k = 0; k < model.nv; ++k)
    {
      v_eps[k] = eps;
      ConfigVectorType q_plus = se3::integrate(model,q,v_eps);
      jac_fd.col(k) = (q_plus - q)/eps;
      v_eps[k] = 0.;
    }
    
    ASSERT_TRUE(jac_row_major.isApprox(jac_fd,sqrt(eps)));
  }
  
  TEST(LocalParameterization,Arm)
  {
    using namespace pinocchio::ceres;
    typedef double Scalar;
    enum { Options = 0 };
    
    typedef ModelLocalParameterization<Scalar,Options> LocalParameterization;
    typedef LocalParameterization::Model Model;
    typedef LocalParameterization::Data Data;
    
    Model model;
    const std::string filename = MODEL_DIRECTORY"/lwr-robot-description/lwr-robot.urdf";
    se3::urdf::buildModel(filename,model);
    
    LocalParameterization local_para(model);
    ASSERT_EQ(local_para.GlobalSize(),model.nq);
    ASSERT_EQ(local_para.LocalSize(),model.nv);
    ASSERT_EQ(model.nq,model.nv);
    
    typedef Model::ConfigVectorType ConfigVectorType;
    typedef Model::TangentVectorType TangentVectorType;
    
    ConfigVectorType q = se3::randomConfiguration(model);
    TangentVectorType v = TangentVectorType::Random(model.nv,1);
    
    ConfigVectorType q_plus_ref = se3::integrate(model,q,v);
    ConfigVectorType q_plus(model.nq,1);
    local_para.Plus(q.data(),v.data(),q_plus.data());
    ASSERT_TRUE(q_plus.isApprox(q_plus_ref));
    
    typedef LocalParameterization::MatrixXs MatrixXs;
    typedef LocalParameterization::RowMatrixXs RowMatrixXs;
    
    RowMatrixXs jac_row_major(model.nq,model.nv); jac_row_major.setZero();
    local_para.ComputeJacobian(q.data(),jac_row_major.data());
    
    MatrixXs jac_fd(model.nq,model.nv);
    
    const Scalar eps = 1e-8;
    TangentVectorType v_eps = TangentVectorType::Zero(model.nv,1);
    for(int k = 0; k < model.nv; ++k)
    {
      v_eps[k] = eps;
      ConfigVectorType q_plus = se3::integrate(model,q,v_eps);
      jac_fd.col(k) = (q_plus - q)/eps;
      v_eps[k] = 0.;
    }
    
    ASSERT_TRUE(jac_row_major.isApprox(jac_fd,sqrt(eps)));
    ASSERT_TRUE(jac_row_major.isIdentity());
  }
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
