############################################################################
# Copyright (c) 2018-2022 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

set(HEADERS_PUBLIC
  TestCUDA_Category.hpp
  TestCUDA_UVM_Category.hpp
  TestHIP_Category.hpp
  TestOPENMP_Category.hpp
  TestOPENMPTARGET_Category.hpp
  TestSERIAL_Category.hpp
  TestSYCL_Category.hpp
  TestTHREADS_Category.hpp
)

add_library(unittest INTERFACE)

target_include_directories(unittest INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/unit_test)

add_library(Cabana::unittest ALIAS unittest)
