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

#ifndef __pinocchio_ceres_constraint_types_hpp
#define __pinocchio_ceres_constraint_types_hpp

namespace pinocchio
{
  namespace ceres
  {
    enum ConstraintTypes
    {
      EQUALITY = 1,
      INEQUALITY = 2,
      BOUNDS = 4
    };
  } // namespace ceres
  
} // namespace pinocchio

#endif // ifndef __pinocchio_ceres_constraint_types_hpp
