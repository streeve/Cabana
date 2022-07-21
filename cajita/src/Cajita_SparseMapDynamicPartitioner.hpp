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
  \file Cajita_SparseMapDynamicPartitioner.hpp
  \brief Multi-node sparse map based dynamic grid partitioner
*/
#ifndef CAJITA_SPARSEMAPDYNAMICPARTITIONER_HPP
#define CAJITA_SPARSEMAPDYNAMICPARTITIONER_HPP

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
  \brief Helper class to set workload for DynamicPartitioner with sparse map.
  \tparam SparseMapType Sparse map type
  \tparam Device Partitioner device type
*/
template <class SparseMapType, typename Device>
class SparseMapWorkloadMeasurer
    : public WorkloadMeasurer<Device>
{
    using memory_space = typename Device::memory_space;
    using execution_space = typename Device::execution_space;

    const SparseMapType& sparseMap;
    MPI_Comm comm;

  public:
    /*!
     \brief Constructor.
     \param sparseMap Sparse map used in workload computation.
     \param comm MPI communicator to use for computing workload.
    */
    SparseMapWorkloadMeasurer( const SparseMapType& sparseMap,
                                                 MPI_Comm comm )
        : sparseMap( sparseMap )
        , comm( comm )
    {
    }

    //! \brief Called by DynamicPartitioner to compute workload
    void compute( Kokkos::View<int***, memory_space>& workload ) override
    {
        Kokkos::parallel_for(
            "compute_local_workload_sparsmap",
            Kokkos::RangePolicy<execution_space>( 0, sparseMap.capacity() ),
            KOKKOS_LAMBDA( uint32_t i ) {
                if ( sparseMap.valid_at( i ) )
                {
                    auto key = sparseMap.key_at( i );
                    int ti, tj, tk;
                    sparseMap.key2ijk( key, ti, tj, tk );
                    Kokkos::atomic_increment(
                        &workload( ti + 1, tj + 1, tk + 1 ) );
                }
            } );
        Kokkos::fence();
        // Wait for other ranks' workload to be ready
        MPI_Barrier( comm );
    }
};

//---------------------------------------------------------------------------//
//! Creation function for SparseMapWorkloadMeasurer from
//! SparseMap
template <typename Device, class SparseMapType>
SparseMapWorkloadMeasurer<SparseMapType, Device>
createSparseMapWorkloadMeasurer(
    const SparseMapType& sparseMap, MPI_Comm comm )
{
    return SparseMapWorkloadMeasurer<SparseMapType, Device>(
        sparseMap, comm );
}

} // end namespace Cajita

#endif // end CAJITA_SPARSEMAPDYNAMICPARTITIONER_HPP
