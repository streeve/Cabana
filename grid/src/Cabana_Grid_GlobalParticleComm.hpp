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

/*!
  \file Cabana_Grid_GlobalParticleComm.hpp
  \brief Global mesh
*/
#ifndef CABANA_GRID_GLOBALPARTICLECOMM_HPP
#define CABANA_GRID_GLOBALPARTICLECOMM_HPP

#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Cabana_Distributor.hpp>
#include <Cabana_Slice.hpp>

#include <memory>
#include <stdexcept>
#include <type_traits>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
/*!
  \brief Global mesh linked cell list.
*/
template <class MemorySpace, class LocalGridType>
class GlobalParticleComm
{
  public:
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = LocalGridType::num_space_dim;
    using mesh_type = typename LocalGridType::mesh_type;
    using global_grid_type = Cabana::Grid::GlobalGrid<mesh_type>;

    using memory_space = MemorySpace;
    using corner_view_type =
        Kokkos::View<double* [num_space_dim][2], memory_space>;
    using destination_view_type = Kokkos::View<int*, memory_space>;
    using rank_view_type = Kokkos::View<int***, memory_space>;

    //! \brief Constructor.
    GlobalParticleComm( const LocalGridType local_grid )
    {
        auto global_grid = local_grid.globalGrid();
        _destinations = destination_view_type(
            Kokkos::ViewAllocateWithoutInitializing( "global_destination" ),
            0 );

        int max_ranks_per_dim = -1;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            _ranks_per_dim[d] = global_grid.dimNumBlock( d );
            if ( _ranks_per_dim[d] > max_ranks_per_dim )
                max_ranks_per_dim = _ranks_per_dim[d];
        }
        copyRanks( global_grid );

        int num_ranks = global_grid.totalNumBlock();
        // Purposely using zero-init. Some entries unused in non-cubic
        // decompositions.
        _local_corners =
            corner_view_type( "local_mpi_boundaries", max_ranks_per_dim );

        _rank_1d = global_grid.blockId();
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _rank[d] = global_grid.dimBlockId( d );

        auto local_mesh = createLocalMesh<memory_space>( local_grid );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            _local_corners( _rank[d], d, 0 ) =
                local_mesh.lowCorner( Cabana::Grid::Own(), d );
            _local_corners( _rank[d], d, 1 ) =
                local_mesh.highCorner( Cabana::Grid::Own(), d );
        }

        // Update local boundaries from all ranks.
        auto comm = global_grid.comm();
        // TODO: Could use subcommunicators instead.
        MPI_Allreduce( MPI_IN_PLACE, _local_corners.data(),
                       _local_corners.size(), MPI_DOUBLE, MPI_SUM, comm );

        Kokkos::Array<int, num_space_dim> double_count;
        double_count[0] = _ranks_per_dim[1] * _ranks_per_dim[2];
        double_count[1] = _ranks_per_dim[0] * _ranks_per_dim[2];
        double_count[2] = _ranks_per_dim[0] * _ranks_per_dim[1];

        for ( std::size_t d = 0; d < num_space_dim; ++d )
            for ( std::size_t r = 0; r < _ranks_per_dim[d]; ++r )
            {
                _local_corners( r, d, 0 ) /= double_count[d];
                _local_corners( r, d, 1 ) /= double_count[d];
            }
    }

    //! Store all cartesian MPI ranks.
    template <class GlobalGridType, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, void> copyRanks( GlobalGridType global_grid )
    {
        Kokkos::resize( _global_ranks, _ranks_per_dim[0], _ranks_per_dim[1],
                        _ranks_per_dim[2] );
        for ( std::size_t i = 0; i < _ranks_per_dim[0]; ++i )
            for ( std::size_t j = 0; j < _ranks_per_dim[1]; ++j )
                for ( std::size_t k = 0; k < _ranks_per_dim[2]; ++k )
                    // Not device accessible (uses MPI), so must be copied.
                    _global_ranks( i, j, k ) =
                        global_grid->blockRank( i, j, k );
    }

    //! Store all cartesian MPI ranks.
    template <class GlobalGridType, std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, void> copyRanks( GlobalGridType global_grid )
    {
        Kokkos::resize( _global_ranks, _ranks_per_dim[0], _ranks_per_dim[1] );
        for ( std::size_t i = 0; i < _ranks_per_dim[0]; ++i )
            for ( std::size_t j = 0; j < _ranks_per_dim[1]; ++j )
                // Not device accessible (uses MPI), so must be copied.
                _global_ranks( i, j ) = global_grid->blockRank( i, j );
    }

    //! Get the MPI rank.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, int>
    getRank( int ijk[num_space_dim] )
    {
        return _global_ranks( ijk[0], ijk[1], ijk[2] );
    }

    //! Get the MPI rank.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, int>
    getRank( int ijk[num_space_dim] )
    {
        return _global_ranks( ijk[0], ijk[1] );
    }

    //! Bin particles across the global grid. Because of MPI partitioning, this
    //! is not a perfect grid (as the Core LinkedCellList is), so we use binary
    //! search instead of direct 3d->1d indexing.
    template <class ExecutionSpace, class PositionType>
    void build( ExecutionSpace exec_space, PositionType positions,
                const std::size_t begin, const std::size_t end )
    {
        Kokkos::Profiling::pushRegion(
            "Cabana::Grid::GlobalParticleComm::build" );

        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );
        assert( end >= begin );
        assert( end <= positions.size() );

        // Must match the size of all particles, even if some can be ignored in
        // this search.
        Kokkos::resize( _destinations, positions.size() );
        // Start with everything staying on this rank.
        Kokkos::deep_copy( _destinations, _rank_1d );

        // Local copies for lambda capture.
        auto local_corners = _local_corners;
        auto ranks_per_dim = _ranks_per_dim;
        auto destinations = _destinations;
        auto build_migrate = KOKKOS_CLASS_LAMBDA( const std::size_t p )
        {
            int ijk[num_space_dim];

            // Find the rank this particle should be moved to.
            for ( std::size_t d = 0; d < num_space_dim; ++d )
            {
                int min = 0;
                int max = ranks_per_dim[d];

                // Check if outside the box in this dimension.
                if ( ( positions( p, d ) < local_corners( min, d, 0 ) ) ||
                     ( positions( p, d ) > local_corners( max - 1, d, 1 ) ) )
                    destinations( p ) = -1;

                // Do a binary search for this particle in this dimension.
                while ( max - min > 1 )
                {
                    int center = Kokkos::floor( ( max + min ) / 2.0 );
                    if ( positions( p, d ) < local_corners( center, d, 0 ) )
                        max = center;
                    else
                        min = center;
                }
                ijk[d] = min;
            }
            // Keep the destination rank for eventual migration.
            destinations( p ) = getRank( ijk );
        };

        Kokkos::RangePolicy<ExecutionSpace> policy( exec_space, begin, end );
        Kokkos::parallel_for( "Cabana::Grid::GlobalParticleComm::build", policy,
                              build_migrate );
        Kokkos::fence();

        Kokkos::Profiling::popRegion();
    }

    template <class ExecutionSpace, class PositionType>
    void build( ExecutionSpace exec_space, PositionType positions )
    {
        build( exec_space, positions, 0, positions.size() );
    }

    template <class PositionType>
    void build( PositionType positions )
    {
        using execution_space = typename memory_space::execution_space;
        // TODO: enable views.
        build( execution_space{}, positions, 0, positions.size() );
    }

    template <class AoSoAType>
    void migrate( MPI_Comm comm, AoSoAType& aosoa )
    {
        Cabana::Distributor<memory_space> distributor( comm, _destinations );
        Cabana::migrate( distributor, aosoa );
    }

  protected:
    int _rank_1d;
    Kokkos::Array<int, num_space_dim> _rank;
    Kokkos::Array<int, num_space_dim> _ranks_per_dim;
    corner_view_type _local_corners;

    rank_view_type _global_ranks;

    destination_view_type _destinations;
};

/*!
  \brief Create global linked cell binning.
  \return Shared pointer to a GlobalParticleComm.
*/
template <class MemorySpace, class LocalGridType>
auto createGlobalParticleComm( const LocalGridType& local_grid )
{
    return std::make_shared<GlobalParticleComm<MemorySpace, LocalGridType>>(
        local_grid );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_GLOBALPARTICLECOMM_HPP
