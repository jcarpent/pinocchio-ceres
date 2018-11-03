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

#include "pinocchio/ceres/utils/crs-matrix.hpp"
#include <random>

Eigen::SparseMatrix<double> generateRadomSparseMatrix(const Eigen::DenseIndex rows,
                                                      const Eigen::DenseIndex cols)
{
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0,1.0);
  
  typedef Eigen::Triplet<double> Triplet;
  std::vector<Triplet> triplets;
  for(int i=0;i<rows;++i)
    for(int j=0;j<cols;++j)
    {
      auto v_ij = dist(gen);                         //generate random number
      if(v_ij < 0.1)
      {
        triplets.push_back(Triplet(i,j,v_ij));      //if larger than treshold, insert it
      }
    }
  Eigen::SparseMatrix<double> mat(rows,cols);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  
  return mat;
}

using namespace pinocchio::ceres::utils;

namespace
{
  
  TEST(CRSMatrix,FromCRSMatrixToEigenSparse)
  {
    typedef Eigen::SparseMatrix<double> SparseMatrixType;
    typedef Eigen::SparseMatrix<double,Eigen::RowMajor> CRSSparseMatrixType;
    Eigen::DenseIndex rows = 5, cols = 8;
    
    SparseMatrixType eigen_ccs_sparse = generateRadomSparseMatrix(rows,cols);
    CRSSparseMatrixType eigen_crs_sparse = eigen_ccs_sparse;
    eigen_crs_sparse.makeCompressed();
    
    ::ceres::CRSMatrix ceres_crs_matrix = SparseMatrixToCRSMatrix(eigen_crs_sparse);
    SparseMatrixType eigen_ccs_sparse_from_ceres_crs = CRSMatrixToSparseMatrix(ceres_crs_matrix);

    ASSERT_TRUE(eigen_ccs_sparse_from_ceres_crs.cols() == eigen_ccs_sparse.cols());
    ASSERT_TRUE(eigen_ccs_sparse_from_ceres_crs.rows() == eigen_ccs_sparse.rows());
    for(Eigen::DenseIndex col_it = 0; col_it < eigen_ccs_sparse.cols(); ++col_it)
    {
      for(Eigen::DenseIndex row_it = 0; row_it < eigen_ccs_sparse.rows(); ++row_it)
      {
        ASSERT_TRUE(eigen_ccs_sparse_from_ceres_crs.coeff(row_it,col_it) == eigen_ccs_sparse.coeff(row_it,col_it));
      }
    }
  }
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

