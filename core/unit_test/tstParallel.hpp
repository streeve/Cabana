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
#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_Parallel.hpp>

#include <gtest/gtest.h>

#include <aosoa_unit_test.hpp>

namespace Test
{

//---------------------------------------------------------------------------//
// Functor work tag for only assigning half the value.
class HalfValueWorkTag
{
};

//---------------------------------------------------------------------------//
// Assignment operator.
template <class AoSoA_t, class SliceType0, class SliceType1, class SliceType2,
          class SliceType3>
class AssignmentOp
{
  public:
    AssignmentOp( AoSoA_t aosoa, float fval, double dval, int ival )
        : _aosoa( aosoa )
        , _slice_0( Cabana::slice<0>( aosoa ) )
        , _slice_1( Cabana::slice<1>( aosoa ) )
        , _slice_2( Cabana::slice<2>( aosoa ) )
        , _slice_3( Cabana::slice<3>( aosoa ) )
        , _fval( fval )
        , _dval( dval )
        , _ival( ival )
        , _dim_1( _slice_0.extent( 2 ) )
        , _dim_2( _slice_0.extent( 3 ) )
        , _dim_3( _slice_0.extent( 4 ) )
    {
    }

    // tagged version that assigns only half the value..
    KOKKOS_INLINE_FUNCTION void operator()( const HalfValueWorkTag&,
                                            const int s, const int a ) const
    {
        // Member 0.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                for ( int k = 0; k < _dim_3; ++k )
                    _slice_0.access( s, a, i, j, k ) =
                        _fval * ( i + j + k ) / 2.0;

        // Member 1.
        _slice_1.access( s, a ) = _ival / 2.0;

        // Member 2.
        for ( int i = 0; i < _dim_1; ++i )
            _slice_2.access( s, a, i ) = _dval * i / 2.0;

        // Member 3.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                _slice_3.access( s, a, i, j ) = _dval * ( i + j ) / 2.0;
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int s, const int a ) const
    {
        // Member 0.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                for ( int k = 0; k < _dim_3; ++k )
                    _slice_0.access( s, a, i, j, k ) = _fval * ( i + j + k );

        // Member 1.
        _slice_1.access( s, a ) = _ival;

        // Member 2.
        for ( int i = 0; i < _dim_1; ++i )
            _slice_2.access( s, a, i ) = _dval * i;

        // Member 3.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                _slice_3.access( s, a, i, j ) = _dval * ( i + j );
    }

  private:
    AoSoA_t _aosoa;
    SliceType0 _slice_0;
    SliceType1 _slice_1;
    SliceType2 _slice_2;
    SliceType3 _slice_3;
    float _fval;
    double _dval;
    int _ival;
    int _dim_1;
    int _dim_2;
    int _dim_3;
};

//---------------------------------------------------------------------------//
// Parallel for test with vectorized indexing.
void runTest2d()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Declare the AoSoA type. Let the library pick an inner array size based
    // on the execution space.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 155;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create a vectorized execution policy using the begin and end of the
    // AoSoA.
    int range_begin = 12;
    int range_end = 135;
    Cabana::SimdPolicy<AoSoA_t::vector_length, TEST_EXECSPACE> policy_1(
        range_begin, range_end );

    // Create a functor to operate on.
    using OpType = AssignmentOp<AoSoA_t, typename AoSoA_t::member_slice_type<0>,
                                typename AoSoA_t::member_slice_type<1>,
                                typename AoSoA_t::member_slice_type<2>,
                                typename AoSoA_t::member_slice_type<3>>;
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    OpType func_1( aosoa, fval, dval, ival );

    // Loop in parallel.
    Cabana::simd_parallel_for( policy_1, func_1, "2d_test_1" );
    Kokkos::fence();

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, range_begin,
                      range_end );

    // Change values and write a second functor.
    fval = 93.4;
    dval = 12.1;
    ival = 4;
    OpType func_2( aosoa, fval, dval, ival );

    // Create another range policy over the entire range.
    Cabana::SimdPolicy<AoSoA_t::vector_length, TEST_EXECSPACE> policy_2(
        0, aosoa.size() );

    // Loop in parallel using 2D array parallelism.
    Cabana::simd_parallel_for( policy_2, func_2, "2d_test_2" );
    Kokkos::fence();

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, 0,
                      aosoa.size() );

    // Change values and write a third functor over a single element.
    fval = 7.7;
    dval = 3.2;
    ival = 9;
    OpType func_3( aosoa, fval, dval, ival );

    // Create another range policy over the entire range.
    range_begin = 16;
    range_end = 17;
    Cabana::SimdPolicy<AoSoA_t::vector_length, TEST_EXECSPACE> policy_3(
        range_begin, range_end );

    // Loop in parallel using 2D array parallelism.
    Cabana::simd_parallel_for( policy_3, func_3, "2d_test_3" );
    Kokkos::fence();

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, range_begin,
                      range_end );

    // Now use the tagged version and assign half the value.
    fval = 93.4;
    dval = 12.1;
    ival = 4;
    OpType func_4( aosoa, fval, dval, ival );

    // Create another range policy over the entire range.
    Cabana::SimdPolicy<AoSoA_t::vector_length, TEST_EXECSPACE, HalfValueWorkTag>
        policy_4( 0, aosoa.size() );

    // Loop in parallel using 2D array parallelism.
    Cabana::simd_parallel_for( policy_4, func_4, "2d_test_4" );
    Kokkos::fence();

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval / 2.0, dval / 2.0, ival / 2.0, dim_1, dim_2,
                      dim_3, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, simd_parallel_for_test ) { runTest2d(); }

//---------------------------------------------------------------------------//

} // end namespace Test
