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

#include "pinocchio/ceres/utils/crs-matrix.hpp"
#include <vector>

namespace pinocchio
{
  namespace ceres
  {
    namespace utils
    {
      
      Eigen::SparseMatrix<double> CRSMatrixToSparseMatrix(const ::ceres::CRSMatrix & crs_matrix)
      {
        typedef Eigen::SparseMatrix<double> ReturnType;
        typedef Eigen::Triplet<double> Triplet;
        
        typedef std::vector<Triplet> TripletVector;
        TripletVector triplets;
        triplets.reserve((size_t)crs_matrix.rows.back());
        
        for(size_t row_it = 0; row_it < crs_matrix.rows.size()-1; ++row_it)
        {
          const int & row_id = crs_matrix.rows[row_it];
          const int & row_id_next = crs_matrix.rows[row_it+1];
          if(row_id == row_id_next) continue; // nothing to add on this row
          
          for(size_t it = (size_t)row_id; it < (size_t)row_id_next; ++it)
          {
            Triplet triplet((ReturnType::StorageIndex)row_it,
                            (ReturnType::StorageIndex)crs_matrix.cols[it],
                            crs_matrix.values[it]);
            triplets.push_back(triplet);
          }
        }
        
        ReturnType res(crs_matrix.num_rows,crs_matrix.num_cols);
        res.setFromTriplets(triplets.begin(),triplets.end());
        return res;
      }
      
      template ::ceres::CRSMatrix SparseMatrixToCRSMatrix<double,Eigen::ColMajor>(const Eigen::SparseMatrix<double,Eigen::ColMajor>&);
      template ::ceres::CRSMatrix SparseMatrixToCRSMatrix<double,Eigen::RowMajor>(const Eigen::SparseMatrix<double,Eigen::RowMajor>&);
      
    } // namespace utils
    
  } // namespace ceres
  
} // namespace pinocchio


