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

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/python.cmake)
INCLUDE(cmake/ide.cmake)
INCLUDE(cmake/apple.cmake)

SET(PROJECT_NAME pinocchio-ceres)
SET(PROJECT_DESCRIPTION "Using Pinocchio inside Google Ceres")
SET(PROJECT_URL "http://github.com/jcarpent/pinocchio-ceres")
OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" ON) 

SET(DOXYGEN_USE_MATHJAX YES)

APPLY_DEFAULT_APPLE_CONFIGURATION()

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

SETUP_PROJECT()

# --- OPTIONS ----------------------------------------
OPTION (INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
OPTION (BUILD_PYTHON_INTERFACE "Build the python binding" ON) 

IF(INITIALIZE_WITH_NAN)
  MESSAGE (STATUS "Initialize with NaN all the Eigen entries.")
  ADD_DEFINITIONS(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
ENDIF(INITIALIZE_WITH_NAN)

# ----------------------------------------------------
# --- DEPENDANCIES -----------------------------------
# ----------------------------------------------------
ADD_REQUIRED_DEPENDENCY("pinocchio >= 1.3.0")
ADD_REQUIRED_DEPENDENCY("eigen3 >= 3.0.5")

FIND_PACKAGE(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
FIND_PACKAGE(Glog REQUIRED)
INCLUDE_DIRECTORIES(${GLOG_INCLUDE_DIRS})

ADD_DEFINITIONS(-DMODEL_DIRECTORY="${${PROJECT_NAME}_SOURCE_DIR}/models")

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
FILE(GLOB_RECURSE ${PROJECT_NAME}_HEADERS 
     ${PROJECT_SOURCE_DIR}/include/pinocchio/ceres/*.hpp 
     ${PROJECT_SOURCE_DIR}/include/pinocchio/ceres/*.hxx 
     )   

ADD_SUBDIRECTORY(src)

# ----------------------------------------------------
# --- EXAMPLES ---------------------------------------
# ----------------------------------------------------
ADD_SUBDIRECTORY(examples)

SETUP_PROJECT_FINALIZE()
