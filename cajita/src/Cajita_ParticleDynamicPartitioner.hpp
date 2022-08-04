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
  \tparam Particles' position positions type (Kokkos::View<Scalar* [3],
  MemorySpace>) \tparam Global grid bottom left corner type \tparam Global grid
  unit cell size type \tparam Partitioner's cell number per tile dim \tparam
  Partitioner's space dim number \tparam Partitioner's device type
*/
template <class PositionViewType, typename ArrayType, typename CellUnit,
          unsigned long long CellPerTileDim, int num_space_dim>
struct ParticleWorkloadFunctor : public WorkloadFunctor
{
    static constexpr unsigned long long cell_bits_per_tile_dim =
        bitCount( CellPerTileDim );

    PositionViewType& _positions;
    int num_particle;
    ArrayType& _global_lower_corner;
    CellUnit _dx;

    /*!
     \brief Constructor.
     \param positions Position of particles used in workload computation.
     \param num_particle The number of particles used in workload computation.
     \param global_lower_corner The bottom-left corner of global grid.
     \param dx The global grid resolution.
    */
    ParticleWorkloadFunctor( const PositionViewType& positions,
                             const int num_particle,
                             const ArrayType& global_low_corner,
                             const CellUnit dx )
    {
        update( positions, num_particle, global_low_corner, dx );
    }

    void update( const PositionViewType& positions, const int num_particle,
                 const ArrayType& global_lower_corner, const CellUnit dx )
    {
        _positions = positions;
        _num_particle = num_particle;
        _global_lower_corner = global_low_corner;
        _dx = dx;
    }

    int workloadSize() { return _num_particle; }

    KOKKOS_INLINE_FUNCTION void
    operator()( Kokkos::View<int***, memory_space>& workload, const int i )
    {
        int ti = static_cast<int>(
                     ( positions( i, 0 ) - lower_corner[0] ) / dx - 0.5 ) >>
                 cell_bits_per_tile_dim;
        int tj = static_cast<int>(
                     ( positions( i, 1 ) - lower_corner[1] ) / dx - 0.5 ) >>
                 cell_bits_per_tile_dim;
        int tz = static_cast<int>(
                     ( positions( i, 2 ) - lower_corner[2] ) / dx - 0.5 ) >>
                 cell_bits_per_tile_dim;
        Kokkos::atomic_increment( &workload( ti + 1, tj + 1, tz + 1 ) );
    }
};

//---------------------------------------------------------------------------//
//! Creation function for ParticleWorkloadFunctor
template <unsigned long long CellPerTileDim, int num_space_dim,
          class PositionViewType, typename ArrayType, typename CellUnit>
auto createParticleDynamicPartitionerWorkloadFunctor(
    const PositionViewType& positions, int num_particle,
    const ArrayType& global_lower_corner, const CellUnit dx )
{
    return ParticleDynamicPartitionerWorkloadFunctor<
        PositionViewType, ArrayType, CellUnit, CellPerTileDim, num_space_dim>(
        positions, num_particle, global_lower_corner, dx );
}

//---------------------------------------------------------------------------//
//! Creation function for 3D with one cell per tile (particles in dense grid).
template <class PositionViewType, typename ArrayType, typename CellUnit>
auto createParticleDynamicPartitionerWorkloadFunctor(
    const PositionViewType& positions, int num_particle,
    const ArrayType& global_lower_corner, const CellUnit dx )
{
    return ParticleDynamicPartitionerWorkloadFunctor<PositionViewType,
                                                     ArrayType, CellUnit, 1, 3>(
        positions, num_particle, global_lower_corner, dx );
}

} // end namespace Cajita

#endif // end CAJITA_PARTICLEDYNAMICPARTITIONER_HPP
