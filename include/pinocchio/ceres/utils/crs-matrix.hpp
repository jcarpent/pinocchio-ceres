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

#ifndef __pinocchio_ceres_utils_crs_matrix_hpp__
#define __pinocchio_ceres_utils_crs_matrix_hpp__

#include <ceres/crs_matrix.h>
#include <Eigen/Sparse>

namespace pinocchio
{
  namespace ceres
  {
    namespace utils
    {
      
      ///
      /// \brief Build an Eigen::SparseMatrix from a ::ceres::CRSMatrix.
      ///
      /// \param[in] crs_matrix The ::ceres::CRSMatrix to convert.
      ///
      /// \returns the converted matrix into Eigen::SparseMatrix<double>.
      ///
      Eigen::SparseMatrix<double> CRSMatrixToSparseMatrix(const ::ceres::CRSMatrix & crs_matrix);
      
      ///
      /// \brief Build a ::ceres::CRSMatrix from an Eigen::SparseMatrix.
      ///
      /// \param[in] eigen_sparse_matrix The Eigen::SparseMatrix to convert.
      ///
      /// \returns the converted matrix into a ceres::CRSMatrix.
      ///
      template<typename Scalar, int Options>
      ::ceres::CRSMatrix SparseMatrixToCRSMatrix(const Eigen::SparseMatrix<Scalar,Options> & eigen_sparse_matrix)
      {
        typedef Eigen::SparseMatrix<Scalar,Options> InputType;
        ::ceres::CRSMatrix ceres_crs_matrix;
        ceres_crs_matrix.num_rows = (int)eigen_sparse_matrix.rows();
        ceres_crs_matrix.num_cols = (int)eigen_sparse_matrix.cols();
        
        const int nnz = eigen_sparse_matrix.outerIndexPtr()[eigen_sparse_matrix.outerSize()];
        
        typedef std::vector<int> IntVector;
        IntVector & outerVector = ceres_crs_matrix.rows;
        IntVector & innerVector = ceres_crs_matrix.cols;
        typedef std::vector<double> DoubleVector;
        DoubleVector & valuesVector = ceres_crs_matrix.values;
        if((Options & Eigen::RowMajorBit) == Eigen::ColMajor)
        {
          outerVector = ceres_crs_matrix.cols;
          innerVector = ceres_crs_matrix.rows;
        }
        valuesVector.reserve((size_t)nnz);
        
        outerVector.reserve((size_t)eigen_sparse_matrix.outerSize());
        innerVector.reserve((size_t)nnz);
        
        for(Eigen::DenseIndex outer_it = 0; outer_it < eigen_sparse_matrix.outerSize(); ++outer_it)
        {
          const typename InputType::StorageIndex & row_id = eigen_sparse_matrix.outerIndexPtr()[outer_it];
          outerVector.push_back((int)row_id);
          const typename InputType::StorageIndex & row_id_next = eigen_sparse_matrix.outerIndexPtr()[outer_it+1];
          if(row_id == row_id_next) continue;
          
          for(typename InputType::StorageIndex col_it = row_id; col_it < row_id_next; ++col_it)
          {
            innerVector.push_back((int)eigen_sparse_matrix.innerIndexPtr()[col_it]);
            valuesVector.push_back((double)eigen_sparse_matrix.valuePtr()[col_it]);
          }
          
        }
        outerVector.push_back((int)eigen_sparse_matrix.outerIndexPtr()[eigen_sparse_matrix.outerSize()]);
        
        return ceres_crs_matrix;
      }
      
      extern template ::ceres::CRSMatrix SparseMatrixToCRSMatrix<double,Eigen::ColMajor>(const Eigen::SparseMatrix<double,Eigen::ColMajor>&);
      extern template ::ceres::CRSMatrix SparseMatrixToCRSMatrix<double,Eigen::RowMajor>(const Eigen::SparseMatrix<double,Eigen::RowMajor>&);
      
    } // namespace utils
    
  } // namespace ceres
  
} // namespace pinocchio

#endif // ifndef __pinocchio_ceres_utils_crs_matrix_hpp__
