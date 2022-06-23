/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_ManualPartitioner.hpp>

#include <mpi.h>

auto createGrid( const Cajita::ManualBlockPartitioner<3>& partitioner,
                 const std::array<bool, 3>& is_periodic,
                 const double cell_size )
{
    // Create the global grid.
    std::array<int, 3> global_num_cell = { 18, 15, 9 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_periodic, partitioner );
    return global_grid;
}

auto createGrid( const Cajita::ManualBlockPartitioner<2>& partitioner,
                 const std::array<bool, 2>& is_periodic,
                 const double cell_size )
{
    // Create the global grid.
    std::array<int, 2> global_num_cell = { 24, 17 };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_periodic, partitioner );
    return global_grid;
}
