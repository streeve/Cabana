/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>

#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

namespace Test
{

//---------------------------------------------------------------------------//
void testMigrate()
{
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    // Create the global mesh.
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    // Create the global grid.
    Cabana::Grid::ManualBlockPartitioner<3> partitioner( ranks_per_dim );
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create a local grid
    int halo_width = 1;
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

    // Create the linked cell list.
    auto linked_cell =
        Cabana::Grid::createLinkedCell<TEST_MEMSPACE>( *local_grid );

    // Create random particles.
    using PoolType = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    using RandomType = Kokkos::Random_XorShift64<ExecutionSpace>;
    PoolType pool( seed );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
            positions( p, d ) = Kokkos::rand<RandomType, double>::draw(
                gen, kokkos_min[d], kokkos_max[d] );
        pool.free_state( gen );
    };
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( mesh, periodic_3d_test ) { testMigrate(); }

} // namespace Test
