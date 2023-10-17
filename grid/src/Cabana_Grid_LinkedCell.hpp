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
  \file Cabana_Grid_LinkedCell.hpp
  \brief Global mesh
*/
#ifndef CABANA_GRID_LINKEDCELL_HPP
#define CABANA_GRID_LINKEDCELL_HPP

#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Cabana_Distributor.hpp>
#include <Cabana_Slice.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
/*!
  \brief Global mesh linked cell list.
*/
template <class MemorySpace, class LocalGridType>
class LinkedCell
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

    //! \brief Constructor.
    LinkedCell( const LocalGridType local_grid )
    {
        _global_grid =
            std::make_shared<global_grid_type>( local_grid.globalGrid() );
        _destinations = destination_view_type(
            Kokkos::ViewAllocateWithoutInitializing( "global_destination" ),
            0 );

        int num_ranks = _global_grid->totalNumBlock();
        // Purposely using zero-init.
        local_corners = corner_view_type( "local_mpi_boundaries", num_ranks );

        for ( std::size_t d = 0; d < num_space_dim; ++d )
            ranks_per_dim[d] = _global_grid->dimNumBlock( d );

        auto rank = _global_grid->blockId();
        auto local_mesh = createLocalMesh<memory_space>( local_grid );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            local_corners( rank, d, 0 ) =
                local_mesh.lowCorner( Cabana::Grid::Own(), d );
            local_corners( rank, d, 1 ) =
                local_mesh.highCorner( Cabana::Grid::Own(), d );
        }

        // Update local boundaries from all ranks.
        auto comm = _global_grid->comm();
        MPI_Allreduce( MPI_IN_PLACE, local_corners.data(), local_corners.size(),
                       MPI_DOUBLE, MPI_SUM, comm );
    }

    KOKKOS_INLINE_FUNCTION
    auto binarySearch( const double px[num_space_dim] )
    {
        int ijk[num_space_dim];
        // Maybe the first guess should be that it stays here?

        // Find the rank this particle should be moved to.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            int min = 0;
            int max = ranks_per_dim[d];
            while ( min < max )
            {
                int center = Kokkos::floor( min + ( max - min ) / 2.0 );
                if ( px[d] > local_corners( center, d, 0 ) )
                    min = center + 1;
                if ( px[d] < local_corners( center, d, 0 ) )
                    max = center - 1;
            }
            ijk[d] = min;
        }
        return _global_grid->blockRank( ijk[0], ijk[1], ijk[2] );
    }

    //! Bin particles across the global grid. Because of MPI partitioning, this
    //! is not a perfect grid (as the Core LinkedCellList is), so we use binary
    //! search instead of direct 3d->1d indexing.
    template <class ExecutionSpace, class PositionType>
    void build( ExecutionSpace exec_space, PositionType positions,
                const std::size_t begin, const std::size_t end )
    {
        Kokkos::Profiling::pushRegion( "Cabana::Grid::LinkedCellList::build" );

        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );
        assert( end >= begin );
        assert( end <= positions.size() );

        Kokkos::resize( _destinations, end - begin );

        auto build_migrate = KOKKOS_LAMBDA( const std::size_t p )
        {
            double px[num_space_dim];
            for ( std::size_t d = 0; d < num_space_dim; ++d )
                px[d] = positions( p, d );

            _destinations( p ) = binarySearch( px );
            std::cout << _destinations( p ) << std::endl;
        };

        Kokkos::RangePolicy<ExecutionSpace> policy( exec_space, begin, end );
        Kokkos::parallel_for( "Cabana::Grid::LinkedCellList::build", policy,
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
        std::cout << "build" << std::endl;
        using execution_space = typename memory_space::execution_space;
        // TODO: enable views.
        build( execution_space{}, positions, 0, positions.size() );
    }

    template <class AoSoAType>
    void migrate( AoSoAType& aosoa )
    {
        Cabana::Distributor<memory_space> distributor( _global_grid->comm(),
                                                       _destinations );
        Cabana::migrate( distributor, aosoa );
    }

  protected:
    Kokkos::Array<int, num_space_dim> ranks_per_dim;
    corner_view_type local_corners;

    std::shared_ptr<global_grid_type> _global_grid;

    destination_view_type _destinations;
};

/*!
  \brief Create global linked cell binning.
  \return Shared pointer to a LinkedCell.
*/
template <class MemorySpace, class LocalGridType>
auto createLinkedCell( const LocalGridType& local_grid )
{
    return std::make_shared<LinkedCell<MemorySpace, LocalGridType>>(
        local_grid );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_LINKEDCELL_HPP
