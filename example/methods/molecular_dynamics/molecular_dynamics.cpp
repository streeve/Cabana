/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <array>
#include <math.h>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

struct Force : public Cabana::Vector<double, 3>
{
};
struct Velocity : public Cabana::Vector<double, 3>
{
};

void molecularDynamics()
{
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;

    // set up block decomposition
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    // Create the global mesh
    double cell_size = 1e-5;
    std::array<double, 3> global_low_corner = { 0, 0, 0 };
    std::array<double, 3> global_high_corner = { 1, 1, 1 };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Create the global grid to create particles from.
    Cajita::DimBlockPartitioner<3> partitioner;
    std::array<bool, 3> periodic = { true, true, true };
    auto global_grid =
        createGlobalGrid( MPI_COMM_WORLD, global_mesh, periodic, partitioner );

    // Create a local grid with halo region.
    unsigned halo_width = 1;
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    auto create = KOKKOS_LAMBDA( const int i )
    {
        for ( int d = 0; d < 3; ++d )
            get( p, Force(), d ) = 0.0;
    };
    Cajita::createParticles( Cabana::InitUniform(), exec_space(), 1, create,
                             particles, local_grid );

    // Simulation details.
    double dt = 1.0;
    int steps = 100;

    // Create the distributor.
    Cajita::ParticleGridMigrate migrate();

    // Create the halo.
    Cajita::ParticleGridHalo gather();

    // Create a neighbor list.
    Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
                       Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    neighbors( positions, 0, positions.size(), cutoff, 1.0, local_low_corner,
               local_high_corner );

    auto first_half = KOKKOS_LAMBDA( const int i )
    {
        const double half_dt_m = half_dt / mass( i );
        v( i, 0 ) += half_dt_m * f( i, 0 );
        v( i, 1 ) += half_dt_m * f( i, 1 );
        v( i, 2 ) += half_dt_m * f( i, 2 );
        x( i, 0 ) += dt * v( i, 0 );
        x( i, 1 ) += dt * v( i, 1 );
        x( i, 2 ) += dt * v( i, 2 );
    };

    auto second_half = KOKKOS_LAMBDA( const int i )
    {
        const double half_dt_m = half_dt / mass( i );
        v( i, 0 ) += half_dt_m * f( i, 0 );
        v( i, 1 ) += half_dt_m * f( i, 1 );
        v( i, 2 ) += half_dt_m * f( i, 2 );
    }

    auto lennard_jones = KOKKOS_LAMBDA( const int i, const int j, double& E )
    {
        double fxi = 0.0;
        double fyi = 0.0;
        double fzi = 0.0;

        const double dx = x( i, 0 ) - x( j, 0 );
        const double dy = x( i, 1 ) - x( j, 1 );
        const double dz = x( i, 2 ) - x( j, 2 );

        const double rsq = dx * dx + dy * dy + dz * dz;
        if ( rsq < cutsq_ij )
        {
            double r2inv = 1.0 / rsq;
            double r6inv = r2inv * r2inv * r2inv;
            double fpair = ( r6inv * ( lj1_ij * r6inv - lj2_ij ) ) * r2inv;
            fxi += dx * fpair;
            fyi += dy * fpair;
            fzi += dz * fpair;
        }

        f( i, 0 ) += fxi;
        f( i, 1 ) += fyi;
        f( i, 2 ) += fzi;
    }

    Kokkos::RangePolicy<exe_space> policy( 0, N_local );

    // Timestep loop.
    double total_energy = 0.0;
    for ( int step = 0; step < numSteps; ++step )
    {
        Kokkos::parallel_for( "VelocityVerletFirstHalf", policy, first_half );

        migrate.apply();

        gather.apply();

        // Create new neighbors every step for convenience.
        neighbors.build( positions, 0, positions.size(), cutoff, 1.0,
                         local_low_corner, local_high_corner );

        Cabana::neighbor_parallel_reduce(
            policy, lennard_jones, neighbors, Cabana::FirstNeighborsTag(),
            Cabana::SerialNeighTag(), total_energy,
            "ForceLJCabanaNeigh::compute_full" );

        Kokkos::parallel_for( "VelocityVerletSecondHalf", policy, second_half );
    }

    // Write the final state.
    Cabana::Experimental::HDF5ParticleOutput::writeTimeStep();
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        molecularDynamics();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
