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

#ifndef CABANA_GRID_LINKEDCELLLIST_HPP
#define CABANA_GRID_LINKEDCELLLIST_HPP

#include <Cabana_LinkedCellList.hpp>
#include <Cabana_Slice.hpp>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
/*!
  \brief Create a LinkedCellList for the local domain.
  \param positions Slice of positions.
  \param begin The beginning index of the AoSoA range to sort.
  \param end The end index of the AoSoA range to sort.
  \param grid_delta Grid size in each dimension.
*/
template <class PositionType, class LocalGridType>
auto createLinkedCellList(
    const PositionType positions, const std::size_t begin,
    const std::size_t end, const LocalGridType local_grid,
    const double grid_delta,
    typename std::enable_if<( is_slice<SliceType>::value ), int>::type* = 0 )
{
    Kokkos::Array<double, 3> delta = { grid_delta, grid_delta, grid_delta };
    auto local_mesh = Cajita::createLocalMesh<memory_space>( *local_grid );
    auto min = local_mesh.lowCorner( Cajita::Ghost() );
    auto max = local_mesh.highCorner( Cajita::Ghost() );

    using linked_cell_type =
        Cabana::LinkedCellList<typename PositionType::memory_space>;
    return linked_cell_type( positions, begin, end, min, max, delta );
}

/*!
  \brief Create a LinkedCellList for the local domain.
  \param positions Slice of positions.
  \param grid_delta Grid size in each dimension.
*/
template <class PositionType, class LocalGridType>
auto createLinkedCellList(
    const PositionType positions, const LocalGridType local_grid,
    const double grid_delta,
    typename std::enable_if<( is_slice<SliceType>::value ), int>::type* = 0 )
{
    return createLinkedCellList( positions, 0, positions.size(), min, max,
                                 delta );
}

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_LINKEDCELLLIST_HPP
