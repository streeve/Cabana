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
  \tparam Sparse map type
  \tparam Partitioner's device type
*/
template <class SparseMapType>
class SparseMapWorkload : public Workload
{
    using const SparseMapType& _sparse_map;

  public:
    /*!
     \brief Constructor.
     \param sparse_map Sparse map used in workload computation.
     \param comm MPI communicator to use for computing workload.
    */
    SparseMapWorkloadFunctor( const SparseMapType& sparse_map )
        : _sparse_map( sparse_map )
    {
    }

    void update( const SparseMapType& sparse_map ) { _sparse_map = sparse_map; }
    int workloadSize() { return _sparse_map.capacity(); }

    KOKKOS_INLINE_FUNCTION void
    operator()( Kokkos::View<int***, memory_space>& workload, const int i )
    {
        if ( sparse_map.valid_at( i ) )
        {
            auto key = sparse_map.key_at( i );
            int ti, tj, tk;
            sparse_map.key2ijk( key, ti, tj, tk );
            Kokkos::atomic_increment( &workload( ti + 1, tj + 1, tk + 1 ) );
        }
    }
}
};

//---------------------------------------------------------------------------//
//! Creation function for SparseMapWorkloadFunctor
template <class SparseMapType>
auto createSparseMapWorkloadFunctor( const SparseMapType& sparse_map )
{
    return SparseMapWorkloadFunctor<SparseMapType>( sparse_map );
}

} // end namespace Cajita

#endif // end CAJITA_SPARSEMAPDYNAMICPARTITIONER_HPP
