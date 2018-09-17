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

MACRO(ADD_EXAMPLE example_name)
  ADD_EXECUTABLE(${example_name} ${example_name}.cpp) 
  TARGET_LINK_LIBRARIES(${example_name} ${CERES_LIBRARIES})
  TARGET_LINK_LIBRARIES(${example_name} ${GLOG_LIBRARIES})
ENDMACRO()

ADD_EXAMPLE(ceres-hello-world) 
