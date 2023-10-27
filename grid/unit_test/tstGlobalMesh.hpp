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

#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace Test
{

using Cabana::Grid::Dim;

//---------------------------------------------------------------------------//
// Test uniform mesh with cubic cells.
template <class ArrayType>
void uniformTest3D1( ArrayType low_corner, ArrayType high_corner )
{
    double cell_size = 0.05;

    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    std::array<int, 3> num_cell = { 18, 188, 4 };
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size );
}

//---------------------------------------------------------------------------//
// Test uniform mesh with number of cells constructor.
template <class ArrayType, class IntArrayType>
void uniformTest3D2( ArrayType low_corner, ArrayType high_corner,
                     IntArrayType num_cell )
{
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, num_cell );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    double cell_size = 0.05;
    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size );
}

//---------------------------------------------------------------------------//
// test uniform mesh with cells that can have a different size in each
// dimension
template <class ArrayType>
void uniformTest3D3( ArrayType low_corner, ArrayType high_corner,
                     ArrayType cell_size )
{
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    std::array<int, 3> num_cell = { 18, 188, 4 };
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size[d] );
}

//---------------------------------------------------------------------------//
// Test uniform mesh with cubic cells.
template <class ArrayType>
void uniformTest2D1( ArrayType low_corner, ArrayType high_corner )
{
    double cell_size = 0.05;

    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    std::array<int, 2> num_cell = { 18, 188 };
    for ( int d = 0; d < 2; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size );
}

//---------------------------------------------------------------------------//
// Test uniform mesh with number of cells constructor.
template <class ArrayType, class IntArrayType>
void uniformTest2D2( ArrayType low_corner, ArrayType high_corner,
                     IntArrayType num_cell )
{
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, num_cell );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    double cell_size = 0.05;
    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size );
}

//---------------------------------------------------------------------------//
// test uniform mesh with cells that can have a different size in each
// dimension
template <class ArrayType>
void uniformTest2D3( ArrayType low_corner, ArrayType high_corner,
                     ArrayType cell_size )
{
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    std::array<int, 2> num_cell = { 18, 188 };
    for ( int d = 0; d < 2; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size[d] );
}

//---------------------------------------------------------------------------//
// test a non uniform 3d mesh
void nonUniformTest3d()
{
    std::vector<float> i_edge = { -0.3, 0.4, 1.1 };
    std::vector<float> j_edge = { 3.3, 8.1, 9.5, 12.2 };
    std::vector<float> k_edge = { -1.1, -0.9, 0.4, 8.8, 19.3 };

    auto global_mesh =
        Cabana::Grid::createNonUniformGlobalMesh( i_edge, j_edge, k_edge );

    EXPECT_FLOAT_EQ( i_edge.front(), global_mesh->lowCorner( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.front(), global_mesh->lowCorner( Dim::J ) );
    EXPECT_FLOAT_EQ( k_edge.front(), global_mesh->lowCorner( Dim::K ) );

    EXPECT_FLOAT_EQ( i_edge.back(), global_mesh->highCorner( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.back(), global_mesh->highCorner( Dim::J ) );
    EXPECT_FLOAT_EQ( k_edge.back(), global_mesh->highCorner( Dim::K ) );

    EXPECT_FLOAT_EQ( i_edge.back() - i_edge.front(),
                     global_mesh->extent( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.back() - j_edge.front(),
                     global_mesh->extent( Dim::J ) );
    EXPECT_FLOAT_EQ( k_edge.back() - k_edge.front(),
                     global_mesh->extent( Dim::K ) );

    EXPECT_EQ( 2, global_mesh->globalNumCell( Dim::I ) );
    EXPECT_EQ( 3, global_mesh->globalNumCell( Dim::J ) );
    EXPECT_EQ( 4, global_mesh->globalNumCell( Dim::K ) );

    const auto& mesh_i = global_mesh->nonUniformEdge( Dim::I );
    int ni = mesh_i.size();
    for ( int i = 0; i < ni; ++i )
        EXPECT_FLOAT_EQ( i_edge[i], mesh_i[i] );

    const auto& mesh_j = global_mesh->nonUniformEdge( Dim::J );
    int nj = mesh_j.size();
    for ( int j = 0; j < nj; ++j )
        EXPECT_FLOAT_EQ( j_edge[j], mesh_j[j] );

    const auto& mesh_k = global_mesh->nonUniformEdge( Dim::K );
    int nk = mesh_k.size();
    for ( int k = 0; k < nk; ++k )
        EXPECT_FLOAT_EQ( k_edge[k], mesh_k[k] );
}

//---------------------------------------------------------------------------//
// test a non uniform 2d mesh
void nonUniformTest2d()
{
    std::vector<float> i_edge = { -0.3, 0.4, 1.1 };
    std::vector<float> j_edge = { 3.3, 8.1, 9.5, 12.2 };

    auto global_mesh =
        Cabana::Grid::createNonUniformGlobalMesh( i_edge, j_edge );

    EXPECT_FLOAT_EQ( i_edge.front(), global_mesh->lowCorner( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.front(), global_mesh->lowCorner( Dim::J ) );

    EXPECT_FLOAT_EQ( i_edge.back(), global_mesh->highCorner( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.back(), global_mesh->highCorner( Dim::J ) );

    EXPECT_FLOAT_EQ( i_edge.back() - i_edge.front(),
                     global_mesh->extent( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.back() - j_edge.front(),
                     global_mesh->extent( Dim::J ) );

    EXPECT_EQ( 2, global_mesh->globalNumCell( Dim::I ) );
    EXPECT_EQ( 3, global_mesh->globalNumCell( Dim::J ) );

    const auto& mesh_i = global_mesh->nonUniformEdge( Dim::I );
    int ni = mesh_i.size();
    for ( int i = 0; i < ni; ++i )
        EXPECT_FLOAT_EQ( i_edge[i], mesh_i[i] );

    const auto& mesh_j = global_mesh->nonUniformEdge( Dim::J );
    int nj = mesh_j.size();
    for ( int j = 0; j < nj; ++j )
        EXPECT_FLOAT_EQ( j_edge[j], mesh_j[j] );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( mesh, uniform_test_3d_std )
{
    std::array<double, 3> low = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high = { -0.3, 9.5, 1.3 };
    uniformTest3D1( low, high );

    std::array<int, 3> num_cell = { 18, 188, 4 };
    uniformTest3D2( low, high, num_cell );

    std::array<double, 3> cell_size = { 0.05, 0.05, 0.05 };
    uniformTest3D3( low, high, cell_size );
}

TEST( mesh, uniform_test_3d_kokkos )
{
    Kokkos::Array<double, 3> low = { -1.2, 0.1, 1.1 };
    Kokkos::Array<double, 3> high = { -0.3, 9.5, 1.3 };
    uniformTest3D1( low, high );

    Kokkos::Array<int, 3> num_cell = { 18, 188, 4 };
    uniformTest3D2( low, high, num_cell );

    Kokkos::Array<double, 3> cell_size = { 0.05, 0.05, 0.05 };
    uniformTest3D3( low, high, cell_size );
}

TEST( mesh, uniform_test_2d_std )
{
    std::array<double, 2> low = { -1.2, 0.1 };
    std::array<double, 2> high = { -0.3, 9.5 };
    uniformTest2D1( low, high );

    std::array<int, 2> num_cell = { 18, 188 };
    uniformTest2D2( low, high, num_cell );

    std::array<double, 2> cell_size = { 0.05, 0.05 };
    uniformTest2D3( low, high, cell_size );
}

TEST( mesh, uniform_test_2d_kokkos )
{
    Kokkos::Array<double, 2> low = { -1.2, 0.1 };
    Kokkos::Array<double, 2> high = { -0.3, 9.5 };
    uniformTest2D1( low, high );

    Kokkos::Array<int, 2> num_cell = { 18, 188 };
    uniformTest2D2( low, high, num_cell );

    Kokkos::Array<double, 2> cell_size = { 0.05, 0.05 };
    uniformTest2D3( low, high, cell_size );
}

TEST( mesh, non_uniform_test_3d ) { nonUniformTest3d(); }

TEST( mesh, non_uniform_test_2d ) { nonUniformTest2d(); }

//---------------------------------------------------------------------------//

} // end namespace Test
