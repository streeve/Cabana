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

#include <Cabana_ParticleInit.hpp>
#include <Cabana_XyzWriter.hpp>

#include <mpi.h>

#include <gtest/gtest.h>

namespace Test
{

void XyzReadWriteTest()
{
    // Create random particles in a box.
    int num_particle = 300;
    double test_radius = 1.67;
    double box_min = -2.5 * test_radius;
    double box_max = 6.3 * test_radius;
    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE>
        output_particles( "to_file", num_particle );
    auto output_positions = Cabana::slice<0>( output_particles );
    Cabana::createRandomParticles( output_positions, num_particle, box_min,
                                   box_max );

    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    std::string filename = "test_" + std::to_string( my_rank ) + ".xyz";

    std::array<double, 3> box_min_array = { box_min, box_min, box_min };
    std::array<double, 3> box_max_array = { box_max, box_max, box_max };

    // Write each rank to a separate file.
    Cabana::writeXyzTimeStep( filename, 0, 0, box_min_array, box_max_array,
                              output_positions );

    // Read the data back on this rank.
    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE>
        input_particles( "from_file", 0 );
    Cabana::readXyzTimeStep( filename, input_particles );
    auto input_positions = Cabana::slice<0>( input_particles );

    // Check that each particle was written/read correctly.
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            auto diff = input_positions( p, d ) - output_positions( p, d );
            EXPECT_LE( std::abs( diff ), 9e-4 );
        }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( cabana_xyz, read_write ) { XyzReadWriteTest(); }

} // end namespace Test
