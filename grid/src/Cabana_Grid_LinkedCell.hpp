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

#include <Cabana_Grid_Types.hpp>

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

    //! \brief Constructor.
    LinkedCell( const LocalGridType local_grid )
    {
        auto global_grid = local_grid.globalGrid();
        int num_ranks = global_grid.totalNumBlock;
        local_corners.resize( num_ranks );

        auto rank = global_grid.blockId();
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            local_corners( rank, d, 0 ) =
                local_mesh.lowCorner( Cajita::Own(), d );
            local_corners( rank, d, 1 ) =
                local_mesh.highCorner( Cajita::Own(), d );
        }
        // Update local boundaries from all ranks (MPI)
    }

    //! Bin particles across the global grid. Because of MPI partitioning, this
    //! is not a perfect grid (as the Core LinkedCellList is), so we use binary
    //! search instead of direct 3d->1d indexing.
    void build() {}

  protected:
    Kokkos::View<double[2][3]*> local_corners;
};

/*!
  \brief Create global linked cell binning.
  \return Shared pointer to a LinkedCell.
*/
template <class LocalGridType>
auto createLinkedCell( const LocalGridType& local_grid )
{
    return std::make_shared<LinkedCell<LocalGridType>>( local_grid );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_LINKEDCELL_HPP
