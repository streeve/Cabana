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

//---------------------------------------------------------------------------//
void testGather( const int halo_width, const int test_halo_width,
                 const int test_type )
{
    PGCommTestData test_data( false, halo_width );
    auto data_host = test_data.data_host;
    auto local_grid = *( test_data.local_grid );
    int num_data = test_data.num_data;

    using DataTypes = Cabana::MemberTypes<int, double[3]>;
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> initial( "initial", num_data );
    Cabana::deep_copy( initial, data_host );
    auto pos_initial = Cabana::slice<1>( initial );

    // Copy to the device.
    Cabana::AoSoA<DataTypes, TEST_MEMSPACE> data_src( "data_src", num_data );
    Cabana::deep_copy( data_src, data_host );

    if ( test_type == 0 )
    {
        // Do the gather with an AoAoA.
        auto grid_halo = Cabana::createGridHalo(
            local_grid, data_src, std::integral_constant<std::size_t, 1>(),
            test_halo_width );
        gridGather( grid_halo, data_src );

        data_host.resize( data_src.size() );
        Cabana::deep_copy( data_host, data_src );
    }
    else if ( test_type == 1 )
    {
        // Create the halo with a slice.
        auto pos_src = Cabana::slice<1>( data_src );
        auto grid_halo =
            Cabana::createGridHalo( local_grid, pos_src, test_halo_width );

        // Resize (cannot resize slice).
        auto halo = grid_halo.getHalo();
        data_src.resize( halo.numLocal() + halo.numGhost() );
        pos_src = Cabana::slice<1>( data_src );

        // Gather with slice.
        gridGather( grid_halo, pos_src );

        data_host.resize( data_src.size() );
        Cabana::deep_copy( data_host, data_src );
    }

    // Check the results.
    int new_num_data = data_host.size();
    auto pos_host = Cabana::slice<1>( data_host );
    std::cout << num_data << " " << new_num_data << std::endl;
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    // Make sure owned particles haven't changed.
    for ( int i = 0; i < num_data; ++i )
    {
        // std::cout << "1 " << pos_host( i, 0 ) << " " << pos_host( i, 1 ) << "
        // "
        //          << pos_host( i, 2 ) << std::endl;
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( pos_host( i, d ), pos_initial( i, d ) );
    }
    for ( int i = num_data; i < new_num_data; ++i )
    {
        // std::cout << "2 " << pos_host( i, 0 ) << " " << pos_host( i, 1 ) << "
        // "
        //          << pos_host( i, 2 ) << std::endl;
        // Make sure particles haven't moved if within allowable halo mesh.
        if ( test_halo_width < halo_width )
        {
            for ( int d = 0; d < 3; ++d )
                EXPECT_DOUBLE_EQ( pos_host( i, d ), pos_initial( i, d ) );
        }
        else
        {
            bool within_x = true;
            bool within_y = true;
            bool within_z = true;
            // Make sure all ghosts are in halo region in at least one
            // direction.
            if ( pos_host( i, Cajita::Dim::I ) < test_data.lo_x )
            {
                EXPECT_LE( pos_host( i, Cajita::Dim::I ), test_data.lo_x );
                EXPECT_GE( pos_host( i, Cajita::Dim::I ),
                           test_data.lo_x - test_data.ghost_x );
            }
            else if ( pos_host( i, Cajita::Dim::I ) > test_data.hi_x )
            {
                std::cout << my_rank << " " << pos_host( i, 0 ) << " "
                          << pos_host( i, 1 ) << " " << pos_host( i, 2 )
                          << std::endl;
                EXPECT_GE( pos_host( i, Cajita::Dim::I ), test_data.hi_x );
                EXPECT_LE( pos_host( i, Cajita::Dim::I ),
                           test_data.hi_x + test_data.ghost_x );
            }
            else
            {
                within_x = false;
            }
            if ( pos_host( i, Cajita::Dim::J ) < test_data.lo_y )
            {
                EXPECT_LE( pos_host( i, Cajita::Dim::J ), test_data.lo_y );
                EXPECT_GE( pos_host( i, Cajita::Dim::J ),
                           test_data.lo_y - test_data.ghost_y );
            }
            else if ( pos_host( i, Cajita::Dim::J ) > test_data.hi_y )
            {
                EXPECT_GE( pos_host( i, Cajita::Dim::J ), test_data.hi_y );
                EXPECT_LE( pos_host( i, Cajita::Dim::J ),
                           test_data.hi_y + test_data.ghost_y );
            }
            else
            {
                within_y = false;
            }
            if ( pos_host( i, Cajita::Dim::K ) < test_data.lo_z )
            {
                EXPECT_LE( pos_host( i, Cajita::Dim::K ), test_data.lo_z );
                EXPECT_GE( pos_host( i, Cajita::Dim::K ),
                           test_data.lo_z - test_data.ghost_z );
            }
            else if ( pos_host( i, Cajita::Dim::K ) > test_data.hi_z )
            {
                EXPECT_GE( pos_host( i, Cajita::Dim::K ), test_data.hi_z );
                EXPECT_LE( pos_host( i, Cajita::Dim::K ),
                           test_data.hi_z + test_data.ghost_z );
            }
            else
            {
                within_z = false;
            }
            if ( !within_x && !within_y && !within_z )
            {
                FAIL() << "Ghost particle outside ghost region. "
                       << pos_host( i, 0 ) << " " << pos_host( i, 1 ) << " "
                       << pos_host( i, 2 );
            }
        }
    }

    // TODO: check number of ghosts created.
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, periodic_test_gather_aosoa )
{
    for ( int i = 1; i < 2; i++ )
        for ( int j = 1; j < 2; j++ )
            testGather( i, j, 0 );
}
/*
TEST( TEST_CATEGORY, periodic_test_gather_slice )
{
    for ( int i = 1; i < 2; i++ )
        for ( int j = 1; j < 2; j++ )
            testGather( i, j, 1 );
}
*/
