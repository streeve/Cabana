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

#include <Cabana_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef Cabana_ENABLE_MPI
#include <mpi.h>
#endif

namespace Cabana
{
namespace Benchmark
{
//---------------------------------------------------------------------------//
// Local timer. Carries multiple data points (the independent variable in
// the parameter sweep) for each timer to allow for parametric sweeps. Each
// timer can do multiple runs over each data point in the parameter sweep. The
// name of the data point and its values can then be injected into the output
// table.
class Timer
{
  public:
    // Create the timer.
    Timer( const std::string& name, const int num_data )
        : _name( name )
        , _starts( num_data )
        , _data( num_data )
        , _is_stopped( num_data, true )
    {
    }

    // Start the timer for the given data point.
    void start( const int data_point )
    {
        if ( !_is_stopped[data_point] )
            throw std::logic_error( "attempted to start a running timer" );
        _starts[data_point] = std::chrono::high_resolution_clock::now();
        _is_stopped[data_point] = false;
    }

    // Stop the timer at the given data point.
    void stop( const int data_point )
    {
        if ( _is_stopped[data_point] )
            throw std::logic_error( "attempted to stop a stopped timer" );
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> fp_micro =
            now - _starts[data_point];
        _data[data_point].push_back( fp_micro.count() );
        _is_stopped[data_point] = true;
    }

  public:
    std::string _name;
    std::vector<std::chrono::high_resolution_clock::time_point> _starts;
    std::vector<std::vector<double>> _data;
    std::vector<bool> _is_stopped;
};

//---------------------------------------------------------------------------//
// Local output.
// Write timer results. Provide the values of the data points so
// they can be injected into the table.
template <class Scalar>
void outputResults( std::ostream& stream, const std::string& data_point_name,
                    const std::vector<Scalar>& data_point_vals,
                    const Timer& timer )
{
    // Write the data header.
    stream << "\n";
    stream << timer._name << "\n";
    stream << data_point_name << " min max ave"
           << "\n";

    // Write out each data point
    for ( std::size_t n = 0; n < timer._data.size(); ++n )
    {
        if ( !timer._is_stopped[n] )
            throw std::logic_error(
                "attempted to output from a running timer" );

        // Compute the minimum.
        double local_min =
            *std::min_element( timer._data[n].begin(), timer._data[n].end() );

        // Compute the maximum.
        double local_max =
            *std::max_element( timer._data[n].begin(), timer._data[n].end() );

        // Compute the average.
        double local_sum = std::accumulate( timer._data[n].begin(),
                                            timer._data[n].end(), 0.0 );
        double average = local_sum / timer._data[n].size();

        // Output.
        stream << data_point_vals[n] << " " << local_min << " " << local_max
               << " " << average << "\n";
    }
}

//---------------------------------------------------------------------------//
// Parallel output.
// Write timer results on rank 0. Provide the values of the data points so
// they can be injected into the table. This function does collective
// communication.
#ifdef Cabana_ENABLE_MPI
template <class Scalar>
void outputResults( std::ostream& stream, const std::string& data_point_name,
                    const std::vector<Scalar>& data_point_vals,
                    const Timer& timer, MPI_Comm comm )
{
    // Get comm rank;
    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );

    // Get comm size;
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    // Write the data header.
    if ( 0 == comm_rank )
    {
        stream << "\n";
        stream << timer._name << "\n";
        stream << "num_rank " << data_point_name << " min max ave"
               << "\n";
    }

    // Write out each data point
    for ( std::size_t n = 0; n < timer._data.size(); ++n )
    {
        if ( !timer._is_stopped[n] )
            throw std::logic_error(
                "attempted to output from a running timer" );

        // Compute the minimum.
        double local_min =
            *std::min_element( timer._data[n].begin(), timer._data[n].end() );
        double global_min = 0.0;
        MPI_Reduce( &local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm );

        // Compute the maximum.
        double local_max =
            *std::max_element( timer._data[n].begin(), timer._data[n].end() );
        double global_max = 0.0;
        MPI_Reduce( &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm );

        // Compute the average.
        double local_sum = std::accumulate( timer._data[n].begin(),
                                            timer._data[n].end(), 0.0 );
        double average = 0.0;
        MPI_Reduce( &local_sum, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm );
        average /= timer._data[n].size() * comm_size;

        // Output on rank 0.
        if ( 0 == comm_rank )
        {
            stream << comm_size << " " << data_point_vals[n] << " "
                   << global_min << " " << global_max << " " << average << "\n";
        }
    }
}
#endif

//---------------------------------------------------------------------------//

// Generate random exponential (non-uniform) particles.
template <class ExecutionSpace, class PositionType>
void createRandomExponential( ExecutionSpace, PositionType& positions,
                              const std::size_t num_clusters,
                              const std::size_t num_particles_per_cluster,
                              const double box_min, const double box_max,
                              const double lambda )
{
    using PoolType = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    using RandomType = Kokkos::Random_XorShift64<ExecutionSpace>;

    Kokkos::RangePolicy<ExecutionSpace> exec_policy(
        0, num_particles_per_cluster );

    int pool_id = 342343901;
    for ( int c = 0; c < num_clusters; ++c )
    {
        PoolType pool( pool_id );

        // Create a new cluster.
        double center[3];
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
        {
            center[d] =
                Kokkos::rand<RandomType, double>::draw( gen, box_min, box_max );
        }
        pool.free_state( gen );

        // Create particles exponentially close to this cluster center.
        int start = c * num_particles_per_cluster;
        auto random_coord_op = KOKKOS_LAMBDA( const int p )
        {
            auto gen = pool.get_state();

            for ( int d = 0; d < 3; ++d )
            {
                double r;
                bool resample = true;
                while ( resample )
                {
                    double rand = Kokkos::rand<RandomType, double>::draw(
                        gen, box_min, box_max );
                    r = lambda * Kokkos::exp( -lambda * rand );
                    if ( r > box_min && r < box_max )
                        resample = false;
                }
                positions( p + start, d ) = r;
            }
            pool.free_state( gen );
        };
        Kokkos::parallel_for( exec_policy, random_coord_op );
        Kokkos::fence();
        pool_id++;
    }
}

template <class AoSoAType>
void readFile( std::string filename, std::vector<AoSoAType>& aosoas,
               std::vector<double>& x_min, std::vector<double>& x_max,
               std::vector<int> problem_sizes )
{
    std::ifstream file_stream;
    file_stream.open( filename );

    std::string line;
    // header
    for ( int l = 0; l < 5; ++l )
        std::getline( file_stream, line );
    // box
    for ( int l = 0; l < 3; ++l )
    {
        std::getline( file_stream, line );
        std::istringstream ss( line );
        std::vector<double> v;
        double val;
        while ( ss >> val )
        {
            v.push_back( val );
        }
        x_min[l] = v[0];
        x_max[l] = v[1];
    }
    std::getline( file_stream, line );

    aosoas[0].resize( problem_sizes[0] );

    auto host_aosoa =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoas[0] );
    auto x = Cabana::slice<0>( host_aosoa, "position" );
    for ( std::size_t p = 0; p < x.size(); ++p )
    {
        std::getline( file_stream, line );
        std::size_t pos;
        for ( std::size_t n = 0; n < 2; n++ )
        {
            pos = line.find( " " );
            line = line.substr( pos + 1, std::string::npos );
        }
        for ( int d = 0; d < 3; ++d )
        {
            pos = line.find( " " );
            x( p, d ) = std::stod( line.substr( 0, pos ) );
            line = line.substr( pos + 1, std::string::npos );
        }
    }
    Cabana::deep_copy( aosoas[0], host_aosoa );
}

template <class Device, class AoSoAType>
void createParticles( Device, std::vector<AoSoAType>& aosoas,
                      std::vector<double>& x_min, std::vector<double>& x_max,
                      std::vector<int> problem_sizes, double cutoff,
                      bool sort = true )
{
    using exec_space = typename Device::execution_space;

    // Declare problem sizes.
    int num_problem_size = problem_sizes.size();

    // Create aosoas.
    for ( std::size_t p = 0; p < problem_sizes.size(); ++p )
    {
        int num_p = problem_sizes[p];

        // Define problem grid.
        x_min[p] = 0.0;
        x_max[p] = 3 * std::pow( num_p, 1.0 / 3.0 );
        aosoas[p].resize( num_p );
        auto x = Cabana::slice<0>( aosoas[p], "position" );
        std::cout << x.size() << std::endl;
        Cabana::createRandomParticles( exec_space{}, x, num_p, x_min[p],
                                       x_max[p] );

        if ( sort )
        {
            // Sort the particles to make them more realistic, e.g. in an MD
            // simulation. They likely won't be randomly scattered about, but
            // rather will be periodically sorted for spatial locality. Bin them
            // in cells the size of the smallest cutoff distance.
            double sort_delta[3] = { cutoff, cutoff, cutoff };
            double grid_min[3] = { x_min[p], x_min[p], x_min[p] };
            double grid_max[3] = { x_max[p], x_max[p], x_max[p] };
            auto x = Cabana::slice<0>( aosoas[p], "position" );
            Cabana::LinkedCellList<Device> linked_cell_list(
                x, sort_delta, grid_min, grid_max );
            Cabana::permute( linked_cell_list, aosoas[p] );
        }
    }
}

} // end namespace Benchmark
} // end namespace Cabana

//---------------------------------------------------------------------------//
