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
*/
template <class ExecutionSpace, class SparseMapType>
class SparseMapWorkloadMeasurer
    : public WorkloadMeasurer<ExecutionSpace,
                              typename SparseMapType::memory_space>
{
    const SparseMapType& sparseMap;
    MPI_Comm comm;

  public:
    //! Kokkos execution space.
    using execution_space = ExecutionSpace;
    //! Kokkos memory space.
    using memory_space = typename SparseMapType::memory_space;
    //! Workload view type.
    using view_type = Kokkos::View<int***, memory_space>;

    /*!
     \brief Constructor.
     \param sparseMap Sparse map used in workload computation.
     \param comm MPI communicator to use for computing workload.
    */
    SparseMapWorkloadMeasurer( const SparseMapType& sparseMap, MPI_Comm comm )
        : sparseMap( sparseMap )
        , comm( comm )
    {
    }

    //! \brief Called by DynamicPartitioner to compute workload
    void compute( view_type& workload ) override
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
template <class ExecutionSpace, class SparseMapType>
auto createSparseMapWorkloadMeasurer( ExecutionSpace,
                                      const SparseMapType& sparseMap,
                                      MPI_Comm comm )
{
    return SparseMapWorkloadMeasurer<ExecutionSpace, SparseMapType>( sparseMap,
                                                                     comm );
}

} // end namespace Cajita

#endif // end CAJITA_SPARSEMAPDYNAMICPARTITIONER_HPP
