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

# Build a fake library if we are currently using a generator. 
# This will gather include files in a same project.

IF(NOT "${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles")
  SET(FAKE_LIBRARY "${PROJECT_NAME}")
  ADD_LIBRARY(${FAKE_LIBRARY} SHARED EXCLUDE_FROM_ALL ${${PROJECT_NAME}_HEADERS})
  SET_TARGET_PROPERTIES(${FAKE_LIBRARY} PROPERTIES LINKER_LANGUAGE CXX)
  ADD_HEADER_GROUP(${PROJECT_NAME}_HEADERS)
ENDIF()

