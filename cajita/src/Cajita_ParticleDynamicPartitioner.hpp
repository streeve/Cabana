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

/*!
  \file Cajita_ParticleDynamicPartitioner.hpp
  \brief Multi-node particle based dynamic grid partitioner
*/
#ifndef CAJITA_PARTICLEDYNAMICPARTITIONER_HPP
#define CAJITA_PARTICLEDYNAMICPARTITIONER_HPP

#include <Cajita_DynamicPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <vector>

#include <mpi.h>

namespace Cajita
{

//---------------------------------------------------------------------------//
/*!
  \brief Helper class to set workload for DynamicPartitioner with particles.
  \tparam Particles' position view type (Kokkos::View<Scalar* [3], MemorySpace>)
  \tparam Global grid bottom left corner type
  \tparam Global grid unit cell size type
  \tparam Partitioner's cell number per tile dim
  \tparam Partitioner's space dim number
  \tparam Partitioner's device type
*/
template <typename ExecutionSpace, typename PositionType, typename ArrayType,
          typename CellUnit, unsigned long long CellPerTileDim,
          int num_space_dim>
class ParticleWorkloadMeasurer
    : public WorkloadMeasurer<ExecutionSpace,
                              typename PositionType::memory_space>
{
    // FIXME: this only exists here because it was initially within the sparse
    // partitioner. Should be removed when the base DynamicPartitioner is
    // refactored.
    static constexpr unsigned long long cell_bits_per_tile_dim =
        bitCount( CellPerTileDim );

    const PositionType& positions;
    int particle_num;
    const ArrayType& global_lower_corner;
    const CellUnit dx;
    MPI_Comm comm;

  public:
    //! Kokkos execution space.
    using execution_space = ExecutionSpace;
    //! Kokkos memory space.
    using memory_space = typename PositionType::memory_space;
    //! Workload view type.
    using view_type = Kokkos::View<int***, memory_space>;

    /*!
     \brief Constructor.
     \param positions Position of particles used in workload computation.
     \param particle_num The number of particles used in workload computation.
     \param global_lower_corner The bottom-left corner of global grid.
     \param dx The global grid resolution.
     \param comm MPI communicator to use for computing workload.
    */
    ParticleWorkloadMeasurer( const PositionType& positions, int particle_num,
                              const ArrayType& global_lower_corner,
                              const CellUnit dx, MPI_Comm comm )
        : positions( positions )
        , particle_num( particle_num )
        , global_lower_corner( global_lower_corner )
        , dx( dx )
        , comm( comm )
    {
    }

    //! \brief Called by DynamicPartitioner to compute workload
    void compute( view_type& workload ) override
    {
        Kokkos::Array<CellUnit, num_space_dim> lower_corner;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            lower_corner[d] = global_lower_corner[d];
        }

        auto dx_copy = dx;
        auto cell_bits_per_tile_dim_copy = cell_bits_per_tile_dim;
        auto lower_x = lower_corner[0];
        auto lower_y = lower_corner[1];
        auto lower_z = lower_corner[2];
        Kokkos::parallel_for(
            "compute_local_workload_parpos",
            Kokkos::RangePolicy<execution_space>( 0, particle_num ),
            KOKKOS_LAMBDA( const int i ) {
                int ti =
                    static_cast<int>(
                        ( positions( i, 0 ) - lower_x ) / dx_copy - 0.5 ) >>
                    cell_bits_per_tile_dim_copy;
                int tj =
                    static_cast<int>(
                        ( positions( i, 1 ) - lower_y ) / dx_copy - 0.5 ) >>
                    cell_bits_per_tile_dim_copy;
                int tz =
                    static_cast<int>(
                        ( positions( i, 2 ) - lower_z ) / dx_copy - 0.5 ) >>
                    cell_bits_per_tile_dim_copy;
                Kokkos::atomic_increment( &workload( ti + 1, tj + 1, tz + 1 ) );
            } );
        Kokkos::fence();
        // Wait for other ranks' workload to be ready
        MPI_Barrier( comm );
    }
};

//---------------------------------------------------------------------------//
//! Creation function for ParticleWorkloadMeasurer from
//! Kokkos::View<Scalar* [3], MemorySpace>
template <unsigned long long CellPerTileDim, int num_space_dim,
          typename ExecutionSpace, typename PositionType, typename ArrayType,
          typename CellUnit>
auto createParticleWorkloadMeasurer( ExecutionSpace,
                                     const PositionType& positions,
                                     int particle_num,
                                     const ArrayType& global_lower_corner,
                                     const CellUnit dx, MPI_Comm comm )
{
    return ParticleWorkloadMeasurer<ExecutionSpace, PositionType, ArrayType,
                                    CellUnit, CellPerTileDim, num_space_dim>(
        positions, particle_num, global_lower_corner, dx, comm );
}

} // end namespace Cajita

#endif // end CAJITA_PARTICLEDYNAMICPARTITIONER_HPP
