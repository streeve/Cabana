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

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// Check the data given a set of values in an aosoa.
template <class aosoa_type>
void checkDataMembers( aosoa_type aosoa, const float fval, const double dval,
                       const int ival, const int dim_1, const int dim_2,
                       const int dim_3, std::size_t begin = 0,
                       std::size_t end = 0 )
{
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );

    auto slice_0 = Cabana::slice<0>( mirror );
    auto slice_1 = Cabana::slice<1>( mirror );
    auto slice_2 = Cabana::slice<2>( mirror );
    auto slice_3 = Cabana::slice<3>( mirror );

    if ( end == 0 )
        end = aosoa.size();
    for ( std::size_t idx = begin; idx < end; ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    EXPECT_EQ( slice_0( idx, i, j, k ), fval * ( i + j + k ) );

        // Member 1.
        EXPECT_EQ( slice_1( idx ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( slice_2( idx, i ), dval * i );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( slice_3( idx, i, j ), dval * ( i + j ) );
    }
}

//---------------------------------------------------------------------------//
// Initialize data members
template <class aosoa_type>
void initializeDataMembers( aosoa_type aosoa, const float fval,
                            const double dval, const int ival, const int dim_1,
                            const int dim_2, const int dim_3 )
{
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );

    Kokkos::parallel_for(
        "init_members", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int idx ) {
            // Member 0.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        slice_0( idx, i, j, k ) = fval * ( i + j + k );

            // Member 1.
            slice_1( idx ) = ival;

            // Member 2.
            for ( int i = 0; i < dim_1; ++i )
                slice_2( idx, i ) = dval * i;

            // Member 3.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    slice_3( idx, i, j ) = dval * ( i + j );
        } );
    Kokkos::fence();
}
