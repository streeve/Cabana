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

#include "../Cabana_BenchmarkUtils.hpp"

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <vector>

//---------------------------------------------------------------------------//
// FIXME: Only run test if HYPRE is compatible with the memory space. This
// is currently written in this structure because HYPRE only has
// compile-time switches for backends and hence only one can be used at a
// time. Once they have a run-time switch we can use that instead.
template <class MemorySpace>
std::enable_if_t<!HypreIsCompatibleWithMemorySpace<MemorySpace>::value, void>
poissonTest( const std::string&, const std::string&, MemorySpace )
{
}

template <class MemorySpace>
std::enable_if_t<HypreIsCompatibleWithMemorySpace<MemorySpace>::value, void>
poissonTest( const std::string& solver_type, const std::string& precond_type,
             MemorySpace )
{
    // Create the global grid.
    double cell_size = 0.25;
    std::array<bool, 3> is_dim_periodic = { false, false, false };
    std::array<double, 3> global_low_corner = { -1.0, -2.0, -1.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 0.5 };
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner, cell_size );

    // Create the global grid.
    DimBlockPartitioner<3> partitioner;
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_mesh = createLocalGrid( global_grid, 1 );
    auto owned_space = local_mesh->indexSpace( Own(), Cell(), Local() );

    // Create the RHS.
    auto vector_layout = createArrayLayout( local_mesh, 1, Cell() );
    auto rhs = createArray<double, MemorySpace>( "rhs", vector_layout );
    ArrayOp::assign( *rhs, 1.0, Own() );

    // Create the LHS.
    auto lhs = createArray<double, MemorySpace>( "lhs", vector_layout );
    ArrayOp::assign( *lhs, 0.0, Own() );

    HYPRE_Init();
    {
        // Create a solver.
        auto solver = createHypreStructuredSolver<double, MemorySpace>(
            solver_type, *vector_layout );

        // Create a 7-point 3d laplacian stencil.
        std::vector<std::array<int, 3>> stencil = {
            { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
            { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };
        solver->setMatrixStencil( stencil );

        // Create the matrix entries. The stencil is defined over cells.
        auto matrix_entry_layout = createArrayLayout( local_mesh, 7, Cell() );
        auto matrix_entries = createArray<double, MemorySpace>(
            "matrix_entries", matrix_entry_layout );
        auto entry_view = matrix_entries->view();
        Cabana::grid_parallel_for(
            "fill_matrix_entries", owned_space, exec_space{},
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                entry_view( i, j, k, 0 ) = 6.0;
                entry_view( i, j, k, 1 ) = -1.0;
                entry_view( i, j, k, 2 ) = -1.0;
                entry_view( i, j, k, 3 ) = -1.0;
                entry_view( i, j, k, 4 ) = -1.0;
                entry_view( i, j, k, 5 ) = -1.0;
                entry_view( i, j, k, 6 ) = -1.0;
            } );

        solver->setMatrixValues( *matrix_entries );

        // Set the tolerance.
        solver->setTolerance( 1.0e-9 );

        // Set the maximum iterations.
        solver->setMaxIter( 2000 );

        // Set the print level.
        solver->setPrintLevel( 2 );

        // Create a preconditioner.
        if ( "none" != precond_type )
        {
            auto preconditioner =
                createHypreStructuredSolver<double, MemorySpace>(
                    precond_type, *vector_layout, true );
            solver->setPreconditioner( preconditioner );
        }

        // Setup the problem.
        solver->setup();

        // Solve the problem.
        solver->solve( *rhs, *lhs );

        // Create a solver reference for comparison.
        auto lhs_ref =
            createArray<double, MemorySpace>( "lhs_ref", vector_layout );
        ArrayOp::assign( *lhs_ref, 0.0, Own() );

        auto ref_solver = createReferenceConjugateGradient<double, MemorySpace>(
            *vector_layout );
        ref_solver->setMatrixStencil( stencil );
        const auto& ref_entries = ref_solver->getMatrixValues();
        auto matrix_view = ref_entries.view();
        auto global_space = local_mesh->indexSpace( Own(), Cell(), Global() );
        int ncell_i = global_grid->globalNumEntity( Cell(), Dim::I );
        int ncell_j = global_grid->globalNumEntity( Cell(), Dim::J );
        int ncell_k = global_grid->globalNumEntity( Cell(), Dim::K );
        Kokkos::parallel_for(
            "fill_ref_entries",
            createExecutionPolicy( owned_space, exec_space{} ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                int gi =
                    i + global_space.min( Dim::I ) - owned_space.min( Dim::I );
                int gj =
                    j + global_space.min( Dim::J ) - owned_space.min( Dim::J );
                int gk =
                    k + global_space.min( Dim::K ) - owned_space.min( Dim::K );
                matrix_view( i, j, k, 0 ) = 6.0;
                matrix_view( i, j, k, 1 ) = ( gi - 1 >= 0 ) ? -1.0 : 0.0;
                matrix_view( i, j, k, 2 ) = ( gi + 1 < ncell_i ) ? -1.0 : 0.0;
                matrix_view( i, j, k, 3 ) = ( gj - 1 >= 0 ) ? -1.0 : 0.0;
                matrix_view( i, j, k, 4 ) = ( gj + 1 < ncell_j ) ? -1.0 : 0.0;
                matrix_view( i, j, k, 5 ) = ( gk - 1 >= 0 ) ? -1.0 : 0.0;
                matrix_view( i, j, k, 6 ) = ( gk + 1 < ncell_k ) ? -1.0 : 0.0;
            } );

        std::vector<std::array<int, 3>> diag_stencil = { { 0, 0, 0 } };
        ref_solver->setPreconditionerStencil( diag_stencil );
        const auto& preconditioner_entries =
            ref_solver->getPreconditionerValues();
        auto preconditioner_view = preconditioner_entries.view();
        Kokkos::parallel_for(
            "fill_preconditioner_entries",
            createExecutionPolicy( owned_space, exec_space{} ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                preconditioner_view( i, j, k, 0 ) = 1.0 / 6.0;
            } );

        ref_solver->setTolerance( 1.0e-11 );
        ref_solver->setPrintLevel( 1 );
        ref_solver->setup();
        ref_solver->solve( *rhs, *lhs_ref );
    }
    HYPRE_Finalize();
}

void tmp()
{
    poissonTest( "PCG", "none", memory_space{} );
    poissonTest( "GMRES", "none", memory_space{} );
    poissonTest( "BiCGSTAB", "none", memory_space{} );
    poissonTest( "PFMG", "none", memory_space{} );
    poissonTest( "PCG", "Diagonal", memory_space{} );
    poissonTest( "GMRES", "Diagonal", memory_space{} );
    poissonTest( "BiCGSTAB", "Diagonal", memory_space{} );
    poissonTest( "PCG", "Jacobi", memory_space{} );
    poissonTest( "GMRES", "Jacobi", memory_space{} );
    poissonTest( "BiCGSTAB", "Jacobi", memory_space{} );
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    // Initialize environment
    MPI_Init( &argc, &argv );

    // Check arguments.
    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
             First argument - file name for output \n \
             Optional second argument - problem size (small or large) \n \
             \n \
             Example: \n \
             $/: ./HypreStructuredSolverPerformance test_results.txt\n" );

    // Define run sizes.
    std::string run_type = "";
    if ( argc > 2 )
    {
        run_type = argv[2];
    }

    // Declare the grid size per dimension
    // currently, testing 3dims+symmetric
    std::vector<double> grid_sizes_per_dim_per_rank = { 16, 32 };
    if ( run_type == "large" )
    {
        grid_sizes_per_dim_per_rank = { 16, 32, 64, 128 };
    }

    // Get the name of the output file.
    std::string filename = argv[1];

    // Barrier before continuing
    MPI_Barrier( MPI_COMM_WORLD );

    // Get comm rank and size;
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    // Get partitioner
    DimBlockPartitioner<3> partitioner;
    // Get ranks per dimension
    std::array<int, 3> ranks_per_dimension =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0, 0 } );

    // Open the output file on rank 0.
    std::fstream file;

    // Output problem details.
    if ( 0 == comm_rank )
    {
        file.open( filename + "_" + std::to_string( comm_size ),
                   std::fstream::out );
        file << "\n";
        file << "Cabana::Grid HYPRE Performance Benchmark"
             << "\n";
        file << "----------------------------------------------"
             << "\n";
        file << "MPI Ranks: " << comm_size << "\n";
        file << "MPI Cartesian Dim Ranks: (" << ranks_per_dimension[0] << ", "
             << ranks_per_dimension[1] << ", " << ranks_per_dimension[2]
             << ")\n";
        file << "----------------------------------------------"
             << "\n";
        file << "\n";
        file << std::flush;
    }

    Kokkos::initialize( argc, argv );

    // Do everything on the default CPU.
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type = host_exec_space::device_type;
    // Do everything on the default device with default memory.
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;

    // Don't run twice on the CPU if only host enabled.
    if ( !std::is_same<device_type, host_device_type>{} )
    {
        performanceTest<device_type>( file, partitioner,
                                      grid_sizes_per_dim_per_rank,
                                      MPI_COMM_WORLD, "device_" );
    }
    performanceTest<host_device_type>( file, partitioner,
                                       grid_sizes_per_dim_per_rank,
                                       MPI_COMM_WORLD, "host_" );

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
