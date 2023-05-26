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

add_executable(MolecularDynamics molecular_dynamics.cpp)
target_link_libraries(MolecularDynamics core Cajita)
add_test(NAME Cabana_methods_molecular_dynamics COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} MolecularDynamics ${MPIEXEC_POSTFLAGS})
set_tests_properties(Cabana_methods_molecular_dynamics PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
