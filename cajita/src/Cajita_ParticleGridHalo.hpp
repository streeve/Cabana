/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_PARTICLEGRIDHALO_HPP
#define CAJITA_PARTICLEGRIDHALO_HPP

#include <Cabana_Halo.hpp>

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>

#include <Kokkos_Core.hpp>

#include <vector>

namespace Cajita
{

//---------------------------------------------------------------------------//
// Particle Grid Halo
//---------------------------------------------------------------------------//

namespace Impl
{
//! \cond impl

// Functor to determine which particles should be ghosted with Cajita grid.
template <class LocalGridType, class PositionSliceType>
struct HaloIds
{
    Kokkos::Array<bool, 3> _periodic;
    Kokkos::Array<double, 3> _global_low;
    Kokkos::Array<double, 3> _global_high;
    Kokkos::Array<double, 3> _global_extent;

    int _min_halo;
    int _neighbor_rank;

    using device_type = typename PositionSliceType::device_type;
    using pos_value = typename PositionSliceType::value_type;

    using DestinationRankView = typename Kokkos::View<int*, device_type>;
    using ShiftViewType = typename Kokkos::View<pos_value**, device_type>;
    using CountView =
        typename Kokkos::View<int, Kokkos::LayoutRight, device_type,
                              Kokkos::MemoryTraits<Kokkos::Atomic>>;
    CountView _send_count;
    DestinationRankView _destinations;
    DestinationRankView _ids;
    ShiftViewType _shifts;
    PositionSliceType _positions;

    Kokkos::Array<int, 3> _ijk;
    Kokkos::Array<double, 3> _min_coord;
    Kokkos::Array<double, 3> _max_coord;

    HaloIds( const LocalGridType& local_grid,
             const PositionSliceType& positions, const int minimum_halo_width,
             const int max_export_guess )
    {
        _positions = positions;
        _destinations = DestinationRankView(
            Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
            max_export_guess );
        _ids = DestinationRankView(
            Kokkos::ViewAllocateWithoutInitializing( "ids" ),
            max_export_guess );
        _shifts =
            ShiftViewType( Kokkos::ViewAllocateWithoutInitializing( "shifts" ),
                           max_export_guess, 3 );
        _send_count = CountView( "halo_send_count" );

        // Check within the halo width, within the local domain.
        const auto& global_grid = local_grid.globalGrid();
        _periodic = { global_grid.isPeriodic( Cajita::Dim::I ),
                      global_grid.isPeriodic( Cajita::Dim::J ),
                      global_grid.isPeriodic( Cajita::Dim::K ) };
        auto dx =
            local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::I );
        auto dy =
            local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::J );
        auto dz =
            local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::K );
        const auto& global_mesh = global_grid.globalMesh();
        _global_low = {
            global_mesh.lowCorner( Cajita::Dim::I ) + minimum_halo_width * dx,
            global_mesh.lowCorner( Cajita::Dim::J ) + minimum_halo_width * dy,
            global_mesh.lowCorner( Cajita::Dim::K ) + minimum_halo_width * dz };
        _global_high = {
            global_mesh.highCorner( Cajita::Dim::I ) - minimum_halo_width * dx,
            global_mesh.highCorner( Cajita::Dim::J ) - minimum_halo_width * dy,
            global_mesh.highCorner( Cajita::Dim::K ) -
                minimum_halo_width * dz };
        _global_extent = { global_mesh.extent( Cajita::Dim::I ),
                           global_mesh.extent( Cajita::Dim::J ),
                           global_mesh.extent( Cajita::Dim::K ) };

        _min_halo = minimum_halo_width;

        build( local_grid );
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int p ) const
    {
        Kokkos::Array<double, 3> pos = { _positions( p, Cajita::Dim::I ),
                                         _positions( p, Cajita::Dim::J ),
                                         _positions( p, Cajita::Dim::K ) };

        // Check the if particle is both in the owned space
        // and the ghosted space of this neighbor (ignore
        // the current cell).
        if ( ( pos[Cajita::Dim::I] > _min_coord[Cajita::Dim::I] &&
               pos[Cajita::Dim::I] < _max_coord[Cajita::Dim::I] ) &&
             ( pos[Cajita::Dim::J] > _min_coord[Cajita::Dim::J] &&
               pos[Cajita::Dim::J] < _max_coord[Cajita::Dim::J] ) &&
             ( pos[Cajita::Dim::K] > _min_coord[Cajita::Dim::K] &&
               pos[Cajita::Dim::K] < _max_coord[Cajita::Dim::K] ) )
        {
            const std::size_t sc = _send_count()++;
            // If the size of the arrays is exceeded, keep
            // counting to resize and fill next.
            if ( sc < _destinations.extent( 0 ) )
            {
                // Keep the destination MPI rank.
                _destinations( sc ) = _neighbor_rank;
                // Keep the particle ID.
                _ids( sc ) = p;
                // Determine if this ghost particle needs to
                // be shifted through the periodic boundary.
                for ( int d = 0; d < 3; ++d )
                {
                    _shifts( sc, d ) = 0.0;
                    if ( _periodic[d] && _ijk[d] )
                    {
                        if ( pos[d] > _global_high[d] )
                            _shifts( sc, d ) = -_global_extent[d];
                        else if ( pos[d] < _global_low[d] )
                            _shifts( sc, d ) = _global_extent[d];
                    }
                }
            }
        }
    }

    //---------------------------------------------------------------------------//
    // Locate particles within the local grid and determine if any from this
    // rank need to be ghosted to one (or more) of the 26 neighbor ranks,
    // keeping track of destination rank, index in the container, and periodic
    // shift needed (but not yet applied).
    void build( const LocalGridType& local_grid )
    {
        using execution_space = typename PositionSliceType::execution_space;
        const auto& local_mesh =
            Cajita::createLocalMesh<Kokkos::HostSpace>( local_grid );

        auto policy =
            Kokkos::RangePolicy<execution_space>( 0, _positions.size() );

        // Add a ghost if this particle is near the local boundary, potentially
        // for each of the 26 neighbors cells. Do this one neighbor rank at a
        // time so that sends are contiguous.
        auto topology = getTopology( local_grid );
        auto unique_topology = Cabana::Impl::getUniqueTopology( topology );
        for ( std::size_t ar = 0; ar < unique_topology.size(); ar++ )
        {
            int nr = 0;
            for ( int k = -1; k < 2; ++k )
            {
                for ( int j = -1; j < 2; ++j )
                {
                    for ( int i = -1; i < 2; ++i, ++nr )
                    {
                        if ( i != 0 || j != 0 || k != 0 )
                        {
                            const int _neighbor_rank = topology[nr];
                            if ( _neighbor_rank == unique_topology[ar] )
                            {
                                auto sis = local_grid.sharedIndexSpace(
                                    Cajita::Own(), Cajita::Cell(), i, j, k,
                                    _min_halo );
                                const int min_ind_i = sis.min( Cajita::Dim::I );
                                const int min_ind_j = sis.min( Cajita::Dim::J );
                                const int min_ind_k = sis.min( Cajita::Dim::K );
                                Kokkos::Array<int, 3> min_ind = {
                                    min_ind_i, min_ind_j, min_ind_k };
                                const int max_ind_i =
                                    sis.max( Cajita::Dim::I ) + 1;
                                const int max_ind_j =
                                    sis.max( Cajita::Dim::J ) + 1;
                                const int max_ind_k =
                                    sis.max( Cajita::Dim::K ) + 1;
                                Kokkos::Array<int, 3> max_ind = {
                                    max_ind_i, max_ind_j, max_ind_k };

                                local_mesh.coordinates( Cajita::Node(),
                                                        min_ind.data(),
                                                        _min_coord.data() );
                                local_mesh.coordinates( Cajita::Node(),
                                                        max_ind.data(),
                                                        _max_coord.data() );
                                _ijk = { i, j, k };

                                Kokkos::parallel_for( "get_halo_ids", policy,
                                                      *this );
                                Kokkos::fence();
                            }
                        }
                    }
                }
            }
            // Shift periodic coordinates in send buffers.
        }
    }

    void rebuild( const LocalGridType& local_grid )
    {
        // Resize views to actual send sizes.
        int dest_size = _destinations.extent( 0 );
        int dest_count = 0;
        Kokkos::deep_copy( dest_count, _send_count );
        if ( dest_count != dest_size )
        {
            Kokkos::resize( _destinations, dest_count );
            Kokkos::resize( _ids, dest_count );
            Kokkos::resize( _shifts, dest_count, 3 );
        }

        // If original view sizes were exceeded, only counting was done so
        // we need to rerun.
        if ( dest_count > dest_size )
        {
            Kokkos::deep_copy( _send_count, 0 );
            build( local_grid );
        }
    }
};

//---------------------------------------------------------------------------//
// Determine which particles should be ghosted, reallocating and recounting if
// needed.
template <class LocalGridType, class PositionSliceType>
auto getHaloIDs(
    const LocalGridType& local_grid, const PositionSliceType& positions,
    const int min_halo_width, const int max_export_guess,
    typename std::enable_if<Cabana::is_slice<PositionSliceType>::value,
                            int>::type* = 0 )
{
    using device_type = typename PositionSliceType::device_type;

    // Get all 26 neighbor ranks.
    auto topology = Impl::getTopology( local_grid );

    // Determine which particles need to be ghosted to neighbors.
    auto halo_ids = Impl::HaloIds<LocalGridType, PositionSliceType>(
        local_grid, positions, min_halo_width, max_export_guess );

    // Rebuild if needed.
    halo_ids.rebuild( local_grid );

    // Create the Cabana Halo.
    auto halo = Cabana::Halo<device_type>( local_grid.globalGrid().comm(),
                                           positions.size(), halo_ids._ids,
                                           halo_ids._destinations, topology );
    return std::make_pair( halo, halo_ids._shifts );
}

//! \endcond
} // namespace Impl

//---------------------------------------------------------------------------//
/*!
  \class PeriodicShift

  \brief Store periodic shifts for halo communication.

  \tparam DeviceType Device type for which the data for this class will be
  allocated and where parallel execution occurs.

  Ideally this would inherit from Halo (PeriodicHalo), combining the periodic
  shift and halo together. This is not currently done because the
  CommunicationPlan contains std member variables that would be captured on the
  device (warnings with NVCC).
*/
template <class DeviceType>
struct PeriodicShift
{
    //! The periodic shift Kokkos View.
    Kokkos::View<double**, DeviceType> _shifts;

    /*!
      \brief Constructor

      \tparam ShiftViewType The periodic shift Kokkos View type.

      \param shifts The periodic shifts for each element being sent.
    */
    template <class ShiftViewType>
    PeriodicShift( const ShiftViewType shifts )
        : _shifts( shifts )
    {
    }
};

/*!
  \class PeriodicModifyAoSoA

  \brief Modify AoSoA buffer with periodic shifts during gather.

  \tparam DeviceType Device type for which the data for this class will be
  allocated and where parallel execution occurs.
  \tparam PositionIndex Particle position index within the AoSoA.
*/
template <class DeviceType, std::size_t PositionIndex>
struct PeriodicModifyAoSoA : PeriodicShift<DeviceType>
{
    using PeriodicShift<DeviceType>::PeriodicShift;
    using PeriodicShift<DeviceType>::_shifts;

    //! Slice spatial dimension.
    std::size_t dim = _shifts.extent( 1 );

    /*!
      \brief Modify the send buffer with periodic shifts.

      \tparam ViewType The container type for the send buffer.

      \param send_buffer Send buffer of positions being ghosted.
      \param i Particle index.
     */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION void operator()( ViewType& send_buffer,
                                            const int i ) const
    {
        for ( std::size_t d = 0; d < dim; ++d )
            Cabana::get<PositionIndex>( send_buffer( i ), d ) +=
                _shifts( i, d );
    }
};

/*!
  \class PeriodicModifySlice

  \brief Modify slice buffer with periodic shifts during gather.

  \tparam DeviceType Device type for which the data for this class will be
  allocated and where parallel execution occurs.
*/
template <class DeviceType>
struct PeriodicModifySlice : PeriodicShift<DeviceType>
{
    //! Periodic shift type
    using PeriodicShift<DeviceType>::PeriodicShift;
    //! The periodic shift Kokkos View
    using PeriodicShift<DeviceType>::_shifts;

    /*!
      \brief Modify the send buffer with periodic shifts.

      \tparam ViewType The container type for the send buffer.

      \param send_buffer Send buffer of positions being ghosted.
      \param i Particle index.
      \param d Dimension index.
    */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION void operator()( ViewType& send_buffer, const int i,
                                            const int d ) const
    {
        send_buffer( i, d ) += _shifts( i, d );
    }
};

//---------------------------------------------------------------------------//
/*!
  \class ParticleGridHalo

  \brief Store communication Halo and Modify functor.

  \tparam HaloType Halo type.
  \tparam ModifyType Modification functor type.
*/
template <class HaloType, class ModifyType>
class ParticleGridHalo
{
    const HaloType _halo;
    const ModifyType _modify;

  public:
    /*!
      \brief Constructor

      \param halo Halo for gather/scatter.
      \param modify Store and apply buffer modifications.
     */
    ParticleGridHalo( const HaloType& halo, const ModifyType& modify )
        : _halo( halo )
        , _modify( modify )
    {
    }

    /*!
      \brief Return stored halo

      \return Halo for gather/scatter.
    */
    HaloType getHalo() const { return _halo; }
    /*!
      \brief Return stored modification functor.

      \return Modify object to access or apply buffer modifications.
    */
    ModifyType getModify() const { return _modify; }
};

//---------------------------------------------------------------------------//
/*!
  \brief Determine which data should be ghosted on another decomposition, using
  bounds of a Cajita grid and taking periodic boundaries into account. AoSoA
  variant.

  \tparam PositionIndex Particle position index within the AoSoA.
  \tparam LocalGridType Cajita LocalGrid type.
  \tparam ParticleContainer AoSoA type.

  \param local_grid The local grid for creating halo and periodicity.
  \param positions Particle positions.
  \param min_halo_width Number of halo mesh widths to include for ghosting.
  \param max_export_guess The allocation size for halo export ranks, IDs, and
  periodic shifts.

  \return ParticleGridHalo containing Halo and PeriodicModify.
*/
template <int PositionIndex, class LocalGridType, class PositionSliceType>
auto createParticleGridHalo( const LocalGridType& local_grid,
                             const PositionSliceType positions,
                             const int min_halo_width,
                             const int max_export_guess = 0 )
{
    using device_type = typename PositionSliceType::device_type;

    auto pair = Impl::getHaloIDs( local_grid, positions, min_halo_width,
                                  max_export_guess );
    using halo_type = Cabana::Halo<device_type>;
    halo_type halo = pair.first;
    auto shifts = pair.second;

    // Create the functor for modifying the buffer.
    using modify_type = PeriodicModifyAoSoA<device_type, PositionIndex>;
    auto periodic_modify = modify_type( shifts );

    // Return Halo and PeriodicModify together.
    ParticleGridHalo<halo_type, modify_type> grid_halo( halo, periodic_modify );
    return grid_halo;
}

//---------------------------------------------------------------------------//
/*!
  \brief Determine which data should be ghosted on another decomposition, using
  bounds of a Cajita grid and taking periodic boundaries into account. Slice
  variant.

  \tparam LocalGridType Cajita LocalGrid type.
  \tparam PositionSliceType Slice type.

  \param local_grid The local grid for creating halo and periodicity.
  \param positions The position slice.
  \param min_halo_width Number of halo mesh widths to include for ghosting.
  \param max_export_guess The allocation size for halo export ranks, IDs, and
  periodic shifts.

  \return ParticleGridHalo containing Halo and PeriodicModify.
*/
template <class LocalGridType, class PositionSliceType>
auto createParticleGridHalo(
    const LocalGridType& local_grid, const PositionSliceType& positions,
    const int min_halo_width, const int max_export_guess = 0,
    typename std::enable_if<Cabana::is_slice<PositionSliceType>::value,
                            int>::type* = 0 )
{
    using device_type = typename PositionSliceType::device_type;

    auto pair = Impl::getHaloIDs( local_grid, positions, min_halo_width,
                                  max_export_guess );
    using halo_type = Cabana::Halo<device_type>;
    halo_type halo = pair.first;
    auto shifts = pair.second;

    // Create the functor for modifying the buffer.
    using modify_type = PeriodicModifySlice<device_type>;
    auto periodic_modify = modify_type( shifts );

    // Return Halo and PeriodicModify together.
    ParticleGridHalo<halo_type, modify_type> grid_halo( halo, periodic_modify );
    return grid_halo;
}

//---------------------------------------------------------------------------//
/*!
  \brief Gather data from one decomposition and ghosts on another decomposition,
  using the bounds and periodicity of a Cajita grid to determine which particles
  should be copied. AoSoA variant.

  \tparam ParticleGridHaloType ParticleGridHalo type - contained ModifyType must
  have an AoSoA-compatible functor to modify the buffer.
  \tparam ParticleContainer AoSoA type.

  \param grid_halo The communication halo taking into account periodic
  boundaries.
  \param particles The particle AoSoA, containing positions.
*/
template <class ParticleGridHaloType, class ParticleContainer>
void particleGridGather(
    const ParticleGridHaloType grid_halo, ParticleContainer& particles,
    typename std::enable_if<Cabana::is_aosoa<ParticleContainer>::value,
                            int>::type* = 0 )
{
    auto halo = grid_halo.getHalo();
    auto modify = grid_halo.getModify();
    particles.resize( halo.numLocal() + halo.numGhost() );

    gather( halo, particles, modify );
}

//---------------------------------------------------------------------------//
/*!
  \brief Gather data from one decomposition and ghosts on another decomposition,
  using the bounds and periodicity of a Cajita grid to determine which particles
  should be copied. Slice variant.

  \tparam ParticleGridHaloType ParticleGridHalo type - contained ModifyType must
  have a slice-compatible functor to modify the buffer.
  \tparam PositionSliceType Slice type.

  \param grid_halo The communication halo taking into account periodic
  boundaries.
  \param positions The position slice.
*/
template <class ParticleGridHaloType, class PositionSliceType>
void particleGridGather(
    const ParticleGridHaloType grid_halo, PositionSliceType& positions,
    typename std::enable_if<Cabana::is_slice<PositionSliceType>::value,
                            int>::type* = 0 )
{
    auto halo = grid_halo.getHalo();
    auto modify = grid_halo.getModify();

    // Must be resized to match local/ghost externally.
    gather( halo, positions, modify );
}

} // namespace Cajita

#endif // end CAJITA_PARTICLEGRIDHALO_HPP
