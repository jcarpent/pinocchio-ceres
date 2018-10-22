#
# Copyright (c) 2018 INRIA
#
# This file is part of Pinocchio-Ceres
# pinocchio is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# pinocchio is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# pinocchio If not, see
# <http://www.gnu.org/licenses/>.

include(../cmake/gtest.cmake)
include(../cmake/test.cmake)
ADD_GTEST_SUITE()

# ADD_TESTCASE(test_name)
# ------------------------
#
# Define a test named `test_name'.
#
#
MACRO(ADD_TESTCASE test_name dependencies)
  
  SET(exe_test_name "test-${test_name}")
  ADD_UNIT_TEST(${exe_test_name} ${test_name}.cpp)
  TARGET_LINK_LIBRARIES(${exe_test_name} gtest_main)
  TARGET_LINK_LIBRARIES(${exe_test_name} ceres)
  SET_PROPERTY(TARGET ${exe_test_name} PROPERTY CXX_STANDARD 11)

  FOREACH(dep ${dependencies})
    PKG_CONFIG_USE_DEPENDENCY(${exe_test_name} ${dep})
  ENDFOREACH()

ENDMACRO(ADD_TESTCASE)

# Generic optimization
ADD_TESTCASE(proximal-cost "eigen3")

# Pinocchio
ADD_TESTCASE(local-parameterization "eigen3;pinocchio;")
