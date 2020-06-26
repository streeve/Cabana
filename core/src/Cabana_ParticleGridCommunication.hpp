/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_PERIODICCOMM_HPP
#define CABANA_PERIODICCOMM_HPP

#include <Cabana_Distributor.hpp>
#include <Cabana_Halo.hpp>
#include <Cabana_Tuple.hpp>

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>

#include <Kokkos_Core.hpp>

#include <vector>

namespace Cabana
{

namespace Impl
{
//---------------------------------------------------------------------------//
// Of the 27 potential local grids figure out which are in our topology.
// Some of the ranks in this list may be invalid. This needs to be updated
// after computing destination ranks to only contain valid ranks.
template <class LocalGridType>
std::vector<int> getTopology( const LocalGridType &local_grid )
{
    std::vector<int> topology( 27, -1 );
    int nr = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i, ++nr )
                topology[nr] = local_grid.neighborRank( i, j, k );
    return topology;
}

//---------------------------------------------------------------------------//
// Make the topology a list of unique and valid ranks. They must all be valid,
// but not necessarily unique (enforced within CommunicationPlan).
inline std::vector<int> getUniqueTopology( std::vector<int> topology )
{
    auto remove_end = std::remove( topology.begin(), topology.end(), -1 );
    std::sort( topology.begin(), remove_end );
    auto unique_end = std::unique( topology.begin(), remove_end );
    topology.resize( std::distance( topology.begin(), unique_end ) );
    return topology;
}

} // namespace Impl

//---------------------------------------------------------------------------//
// Grid Distributor/migrate
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
\brief Wrap particles through periodic bounds according to Cajita grid global
bounds.

\tparam LocalGridType Cajita LocalGrid type.

\tparam PositionSliceType Particle position type.

\param local_grid The local grid containing periodicity and system bound
information.

\param positions The particle position container, either Slice or View.
*/
template <class LocalGridType, class PositionSliceType>
void periodicWrap( const LocalGridType &local_grid,
                   PositionSliceType &positions )
{
    using execution_space = typename PositionSliceType::execution_space;

    const auto &global_grid = local_grid.globalGrid();
    const auto &global_mesh = global_grid.globalMesh();
    const Kokkos::Array<bool, 3> periodic = {
        global_grid.isPeriodic( Cajita::Dim::I ),
        global_grid.isPeriodic( Cajita::Dim::J ),
        global_grid.isPeriodic( Cajita::Dim::K ) };
    const Kokkos::Array<double, 3> global_low = {
        global_mesh.lowCorner( Cajita::Dim::I ),
        global_mesh.lowCorner( Cajita::Dim::J ),
        global_mesh.lowCorner( Cajita::Dim::K ) };
    const Kokkos::Array<double, 3> global_high = {
        global_mesh.highCorner( Cajita::Dim::I ),
        global_mesh.highCorner( Cajita::Dim::J ),
        global_mesh.highCorner( Cajita::Dim::K ) };
    const Kokkos::Array<double, 3> global_extent = {
        global_mesh.extent( Cajita::Dim::I ),
        global_mesh.extent( Cajita::Dim::J ),
        global_mesh.extent( Cajita::Dim::K ) };
    Kokkos::parallel_for(
        "periodic_wrap",
        Kokkos::RangePolicy<execution_space>( 0, positions.size() ),
        KOKKOS_LAMBDA( const int p ) {
            for ( int d = 0; d < 3; ++d )
            {
                if ( periodic[d] )
                {
                    if ( positions( p, d ) > global_high[d] )
                        positions( p, d ) -= global_extent[d];
                    else if ( positions( p, d ) < global_low[d] )
                        positions( p, d ) += global_extent[d];
                }
            }
        } );
}

//---------------------------------------------------------------------------//
/*!
\brief Check for the number of particles that must be communicated

\tparam LocalGridType Cajita LocalGrid type.

\tparam PositionSliceType Particle position type.

\param local_grid The local grid containing periodicity and system bound
information.

\param positions The particle position container, either Slice or View.

\param minimum_halo_width Number of halo mesh widths to include for ghosting.
*/
template <class LocalGridType, class PositionSliceType>
int migrateCount( const LocalGridType &local_grid,
                  const PositionSliceType &positions,
                  const int minimum_halo_width )
{
    using execution_space = typename PositionSliceType::execution_space;

    // Check within the halo width, within the ghosted domain.
    const auto &local_mesh =
        Cajita::createLocalMesh<Kokkos::HostSpace>( local_grid );
    auto dx = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::I );
    auto dy = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::J );
    auto dz = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::K );
    const Kokkos::Array<double, 3> local_low = {
        local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::I ) +
            minimum_halo_width * dx,
        local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::J ) +
            minimum_halo_width * dy,
        local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::K ) +
            minimum_halo_width * dz };
    const Kokkos::Array<double, 3> local_high = {
        local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::I ) -
            minimum_halo_width * dx,
        local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::J ) -
            minimum_halo_width * dy,
        local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::K ) -
            minimum_halo_width * dz };
    int comm_count = 0;
    Kokkos::parallel_reduce(
        "redistribute_count",
        Kokkos::RangePolicy<execution_space>( 0, positions.size() ),
        KOKKOS_LAMBDA( const int p, int &result ) {
            if ( positions( p, Cajita::Dim::I ) < local_low[Cajita::Dim::I] ||
                 positions( p, Cajita::Dim::I ) > local_high[Cajita::Dim::I] ||
                 positions( p, Cajita::Dim::J ) < local_low[Cajita::Dim::J] ||
                 positions( p, Cajita::Dim::J ) > local_high[Cajita::Dim::J] ||
                 positions( p, Cajita::Dim::K ) < local_low[Cajita::Dim::K] ||
                 positions( p, Cajita::Dim::K ) > local_high[Cajita::Dim::K] )
                result += 1;
        },
        comm_count );

    MPI_Allreduce( MPI_IN_PLACE, &comm_count, 1, MPI_INT, MPI_SUM,
                   local_grid.globalGrid().comm() );

    return comm_count;
}

namespace Impl
{
//---------------------------------------------------------------------------//
// Locate the particles in the local grid and get their destination rank.
// Particles are assumed to only migrate to a location in the 26 neighbor halo
// or stay on this rank. If the particle crosses a global periodic boundary,
// wrap it's coordinates back into the domain.
template <class LocalGridType, class PositionSliceType, class NeighborRankView,
          class DestinationRankView>
void getMigrateDestinations( const LocalGridType &local_grid,
                             const NeighborRankView &neighbor_ranks,
                             DestinationRankView &destinations,
                             const PositionSliceType &positions )
{
    using execution_space = typename PositionSliceType::execution_space;

    const auto &local_mesh =
        Cajita::createLocalMesh<Kokkos::HostSpace>( local_grid );

    // Check within the local domain.
    const Kokkos::Array<double, 3> local_low = {
        local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::I ),
        local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::J ),
        local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::K ) };
    const Kokkos::Array<double, 3> local_high = {
        local_mesh.highCorner( Cajita::Own(), Cajita::Dim::I ),
        local_mesh.highCorner( Cajita::Own(), Cajita::Dim::J ),
        local_mesh.highCorner( Cajita::Own(), Cajita::Dim::K ) };

    Kokkos::parallel_for(
        "get_migrate_destinations",
        Kokkos::RangePolicy<execution_space>( 0, positions.size() ),
        KOKKOS_LAMBDA( const int p ) {
            // Compute the logical index of the neighbor we are sending to.
            int nid[3] = { 1, 1, 1 };
            for ( int d = 0; d < 3; ++d )
            {
                if ( positions( p, d ) < local_low[d] )
                    nid[d] = 0;
                else if ( positions( p, d ) > local_high[d] )
                    nid[d] = 2;
            }

            // Compute the destination MPI rank.
            destinations( p ) = neighbor_ranks(
                nid[Cajita::Dim::I] +
                3 * ( nid[Cajita::Dim::J] + 3 * nid[Cajita::Dim::K] ) );
        } );
    // TODO: fuse kernels
    periodicWrap( local_grid, positions );
}

} // namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief gridDistributor determines which data should be migrated from one
  uniquely-owned decomposition to another uniquely-owned decomposition, using
  bounds of a Cajita grid and taking periodic boundaries into account.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam PositionContainer AoSoA type.

  \param local_grid The local grid containing periodicity and system bound
  information.

  \param positions The particle positions.

  \return Distributor for later migration.
*/
template <class LocalGridType, class PositionSliceType>
Distributor<typename PositionSliceType::device_type>
gridDistributor( const LocalGridType &local_grid, PositionSliceType &positions )
{
    using device_type = typename PositionSliceType::device_type;

    // Get all 26 neighbor ranks.
    auto topology = Impl::getTopology( local_grid );

    Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        neighbor_ranks( topology.data(), topology.size() );
    auto nr_mirror =
        Kokkos::create_mirror_view_and_copy( device_type(), neighbor_ranks );
    Kokkos::View<int *, device_type> destinations(
        Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
        positions.size() );

    // Determine destination ranks for all particles and wrap positions across
    // periodic boundaries.
    Impl::getMigrateDestinations( local_grid, nr_mirror, destinations,
                                  positions );

    // Ensure neighbor ranks are valid and unique.
    auto unique_topology = Impl::getUniqueTopology( topology );

    // Create the Cabana distributor.
    Distributor<device_type> distributor( local_grid.globalGrid().comm(),
                                          destinations, unique_topology );
    return distributor;
}

//---------------------------------------------------------------------------//
/*!
  \brief gridMigrate migrates data from one uniquely-owned decomposition to
  another uniquely-owned decomposition, using the bounds and periodic boundaries
  of a Cajita grid to determine which particles should be moved. In-place
  variant.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam PositionContainer AoSoA type.

  \tparam PositionIndex Particle position index within the AoSoA.

  \param local_grid The local grid containing periodicity and system bound
  information.

  \param particles The particle AoSoA, containing positions.

  \param PositionIndex Particle position index within the AoSoA.

  \param min_halo_width Number of halo mesh widths to allow particles before
  migrating.

  \param force_migrate Migrate particles outside the local domain regardless of
  ghosted halo.
*/
template <class LocalGridType, class ParticleContainer,
          std::size_t PositionIndex>
void gridMigrate( const LocalGridType &local_grid, ParticleContainer &particles,
                  std::integral_constant<std::size_t, PositionIndex>,
                  const int min_halo_width, const bool force_migrate = false )
{
    // Get the positions.
    auto positions = slice<PositionIndex>( particles );

    // When false, this option checks that any particles are nearly outside the
    // ghosted halo region (outside the  min_halo_width) before initiating
    // migration. Otherwise, anything outside the local domain is migrated
    // regardless of position in the halo.
    if ( !force_migrate )
    {
        // Check to see if we need to communicate.
        auto comm_count = migrateCount( local_grid, positions, min_halo_width );

        // If we have no particles near the ghosted boundary, then exit.
        if ( 0 == comm_count )
            return;
    }

    auto distributor = gridDistributor( local_grid, positions );

    // Redistribute the particles.
    migrate( distributor, particles );
}

//---------------------------------------------------------------------------//
/*!
  \brief gridMigrate migrates data from one uniquely-owned decomposition to
  another uniquely-owned decomposition, using the bounds and periodic boundaries
  of a Cajita grid to determine which particles should be moved. Separate AoSoA
  variant.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam ParticleContainer AoSoA type.

  \tparam PositionIndex Particle position index within the AoSoA.

  \param local_grid The local grid containing periodicity and system bound
  information.

  \param src_particles The source particle AoSoA, containing positions.

  \param PositionIndex Particle position index within the AoSoA.

  \param src_particles The destination particle AoSoA, containing positions.

  \param min_halo_width Number of halo mesh widths to allow particles before
  migrating.

  \param force_migrate Migrate particles outside the local domain regardless of
  ghosted halo.
*/
template <class LocalGridType, class ParticleContainer,
          std::size_t PositionIndex>
void gridMigrate( const LocalGridType &local_grid,
                  ParticleContainer &src_particles,
                  std::integral_constant<std::size_t, PositionIndex>,
                  ParticleContainer &dst_particles, const int min_halo_width,
                  const bool force_migrate = false )
{
    // Get the positions.
    auto positions = slice<PositionIndex>( src_particles );

    // When false, this option checks that any particles are nearly outside the
    // ghosted halo region (outside the  min_halo_width) before initiating
    // migration. Otherwise, anything outside the local domain is migrated
    // regardless of position in the halo.
    if ( !force_migrate )
    {
        // Check to see if we need to communicate.
        auto comm_count = migrateCount( local_grid, positions, min_halo_width );

        // If we have no particles near the ghosted boundary, copy, then exit.
        if ( 0 == comm_count )
        {
            Cabana::deep_copy( dst_particles, src_particles );
            return;
        }
    }

    auto distributor = gridDistributor( local_grid, positions );

    // Resize as needed.
    dst_particles.resize( distributor.totalNumImport() );

    // Redistribute the particles.
    migrate( distributor, src_particles, dst_particles );
}

//---------------------------------------------------------------------------//
// Grid Halo/gather
//---------------------------------------------------------------------------//

namespace Impl
{
//---------------------------------------------------------------------------//
// Locate particles within the local grid and determine if any from this rank
// need to be ghosted to one (or more) of the 26 neighbor ranks, keeping track
// of destination rank, index in the container, and periodic shift needed (but
// not yet applied).
template <class LocalGridType, class PositionSliceType, class CountView,
          class DestinationRankView, class ShiftRankView>
void getHaloIds( const LocalGridType &local_grid, CountView &send_count,
                 DestinationRankView &destinations, DestinationRankView &ids,
                 ShiftRankView &shifts, const PositionSliceType &positions,
                 const int minimum_halo_width )
{
    using execution_space = typename PositionSliceType::execution_space;

    // Check within the halo width, within the local domain.
    const auto &global_grid = local_grid.globalGrid();
    const Kokkos::Array<bool, 3> periodic = {
        global_grid.isPeriodic( Cajita::Dim::I ),
        global_grid.isPeriodic( Cajita::Dim::J ),
        global_grid.isPeriodic( Cajita::Dim::K ) };
    auto dx = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::I );
    auto dy = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::J );
    auto dz = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::K );
    const auto &global_mesh = global_grid.globalMesh();
    const Kokkos::Array<double, 3> global_low = {
        global_mesh.lowCorner( Cajita::Dim::I ) + minimum_halo_width * dx,
        global_mesh.lowCorner( Cajita::Dim::J ) + minimum_halo_width * dy,
        global_mesh.lowCorner( Cajita::Dim::K ) + minimum_halo_width * dz };
    const Kokkos::Array<double, 3> global_high = {
        global_mesh.highCorner( Cajita::Dim::I ) - minimum_halo_width * dx,
        global_mesh.highCorner( Cajita::Dim::J ) - minimum_halo_width * dy,
        global_mesh.highCorner( Cajita::Dim::K ) - minimum_halo_width * dz };
    const Kokkos::Array<double, 3> global_extent = {
        global_mesh.extent( Cajita::Dim::I ),
        global_mesh.extent( Cajita::Dim::J ),
        global_mesh.extent( Cajita::Dim::K ) };
    const auto &local_mesh =
        Cajita::createLocalMesh<Kokkos::HostSpace>( local_grid );

    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    auto policy = Kokkos::RangePolicy<execution_space>( 0, positions.size() );

    // Add a ghost if this particle is near the local boundary, potentially
    // for each of the 26 neighbors cells. Do this one neighbor rank at a time
    // so that sends are contiguous.
    auto topology = getTopology( local_grid );
    auto unique_topology = getUniqueTopology( topology );
    // Put this rank first to make sure shifts match CommPlan export_ranks.
    for ( auto &n : unique_topology )
    {
        if ( n == my_rank )
        {
            std::swap( n, unique_topology[0] );
            break;
        }
    }
    for ( std::size_t ar = 0; ar < unique_topology.size(); ar++ )
    {
        // TODO: possible grid_parallel_for
        int nr = 0;
        for ( int k = -1; k < 2; ++k )
            for ( int j = -1; j < 2; ++j )
                for ( int i = -1; i < 2; ++i, ++nr )
                {
                    auto neighbor_rank = topology.at( nr );
                    if ( neighbor_rank == unique_topology.at( ar ) )
                    {
                        auto sis = local_grid.sharedIndexSpace(
                            Cajita::Own(), Cajita::Cell(), i, j, k,
                            minimum_halo_width );
                        const int min_ind_i = sis.min( Cajita::Dim::I );
                        const int min_ind_j = sis.min( Cajita::Dim::J );
                        const int min_ind_k = sis.min( Cajita::Dim::K );
                        Kokkos::Array<int, 3> min_ind = { min_ind_i, min_ind_j,
                                                          min_ind_k };
                        const int max_ind_i = sis.max( Cajita::Dim::I ) + 1;
                        const int max_ind_j = sis.max( Cajita::Dim::J ) + 1;
                        const int max_ind_k = sis.max( Cajita::Dim::K ) + 1;
                        Kokkos::Array<int, 3> max_ind = { max_ind_i, max_ind_j,
                                                          max_ind_k };

                        Kokkos::Array<double, 3> min_coord;
                        Kokkos::Array<double, 3> max_coord;
                        local_mesh.coordinates( Cajita::Node(), min_ind.data(),
                                                min_coord.data() );
                        local_mesh.coordinates( Cajita::Node(), max_ind.data(),
                                                max_coord.data() );

                        auto halo_ids_func = KOKKOS_LAMBDA( const int p )
                        {
                            Kokkos::Array<double, 3> pos = {
                                positions( p, Cajita::Dim::I ),
                                positions( p, Cajita::Dim::J ),
                                positions( p, Cajita::Dim::K ) };

                            // Check the if particle is both in the owned space
                            // and the ghosted space of this neighbor (ignore
                            // the current cell).
                            if ( ( pos[Cajita::Dim::I] >
                                       min_coord[Cajita::Dim::I] &&
                                   pos[Cajita::Dim::I] <
                                       max_coord[Cajita::Dim::I] ) &&
                                 ( pos[Cajita::Dim::J] >
                                       min_coord[Cajita::Dim::J] &&
                                   pos[Cajita::Dim::J] <
                                       max_coord[Cajita::Dim::J] ) &&
                                 ( pos[Cajita::Dim::K] >
                                       min_coord[Cajita::Dim::K] &&
                                   pos[Cajita::Dim::K] <
                                       max_coord[Cajita::Dim::K] ) &&
                                 ( i != 0 || j != 0 || k != 0 ) )
                            {
                                const std::size_t sc = send_count()++;
                                // If the size of the arrays is exceeded, keep
                                // counting to resize and fill next.
                                if ( sc < destinations.extent( 0 ) )
                                {
                                    // Keep the destination MPI rank.
                                    destinations( sc ) = neighbor_rank;
                                    // Keep the particle ID.
                                    ids( sc ) = p;

                                    // Determine if this ghost particle needs to
                                    // be shifted through the periodic boundary.
                                    const Kokkos::Array<int, 3> ijk = { i, j,
                                                                        k };
                                    for ( int d = 0; d < 3; ++d )
                                    {
                                        shifts( sc, d ) = 0.0;
                                        if ( periodic[d] && ijk[d] )
                                        {
                                            if ( pos[d] > global_high[d] )
                                                shifts( sc, d ) =
                                                    -global_extent[d];
                                            else if ( pos[d] < global_low[d] )
                                                shifts( sc, d ) =
                                                    global_extent[d];
                                        }
                                    }
                                }
                            }
                        };
                        Kokkos::parallel_for( "get_halo_ids", policy,
                                              halo_ids_func );
                    }
                }
    }
    // Shift periodic coordinates in send buffers.
}

} // namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief gridHalo determines which data should be ghosted on another
  decomposition, using bounds of a Cajita grid and taking periodic boundaries
  into account.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam PositionSliceType Slice/View type.

  \param local_grid The local grid containing periodicity and system bound
  information.

  \param positions The particle positions.

  \param min_halo_width Number of halo mesh widths to include for ghosting.

  \param max_export_guess The allocation size for halo export ranks, IDs, and
  periodic shifts

  \return pair A std::pair containing the Halo and periodic shift array for
  a later gather.
*/
template <class LocalGridType, class PositionSliceType>
auto gridHalo( const LocalGridType &local_grid, PositionSliceType &positions,
               const int min_halo_width, const int max_export_guess = 0 )
{
    using device_type = typename PositionSliceType::device_type;
    using pos_value = typename PositionSliceType::value_type;

    // Get all 26 neighbor ranks.
    auto topology = Impl::getTopology( local_grid );

    Kokkos::View<int *, device_type> destinations(
        Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
        max_export_guess );
    Kokkos::View<int *, device_type> ids(
        Kokkos::ViewAllocateWithoutInitializing( "ids" ), max_export_guess );
    Kokkos::View<pos_value **, device_type> shifts(
        Kokkos::ViewAllocateWithoutInitializing( "shifts" ), max_export_guess,
        3 );
    Kokkos::View<int, Kokkos::LayoutRight, device_type,
                 Kokkos::MemoryTraits<Kokkos::Atomic>>
        send_count( "halo_send_count" );

    // Determine which particles need to be ghosted to neighbors.
    Impl::getHaloIds( local_grid, send_count, destinations, ids, shifts,
                      positions, min_halo_width );

    // Resize views to actual send sizes.
    int dest_size = destinations.extent( 0 );
    int dest_count = 0;
    Kokkos::deep_copy( dest_count, send_count );
    if ( dest_count != dest_size )
    {
        Kokkos::resize( destinations, dest_count );
        Kokkos::resize( ids, dest_count );
        Kokkos::resize( shifts, dest_count, 3 );
    }

    // If original view sizes were exceeded, only counting was done so we
    // need to rerun.
    if ( dest_count > dest_size )
    {
        Kokkos::deep_copy( send_count, 0 );
        Impl::getHaloIds( local_grid, send_count, destinations, ids, shifts,
                          positions, min_halo_width );
    }

    // Ensure neighbor ranks are valid and unique.
    auto unique_topology = Impl::getUniqueTopology( topology );

    // Create the Cabana Halo.
    auto halo =
        Halo<device_type>( local_grid.globalGrid().comm(), positions.size(),
                           ids, destinations, unique_topology );
    return std::make_pair( halo, shifts );
}

//---------------------------------------------------------------------------//
/*!
  \class PeriodicHalo

  \brief Store information for periodic halo communication.
*/
template <class HaloType, class ShiftViewType, class ParticleContainer,
          std::size_t PositionIndex>
struct PeriodicHalo
{
    using TupleType = typename ParticleContainer::tuple_type;

    const HaloType _halo;
    const ShiftViewType _shifts;

    /*!
      \brief Constructor.

      \param pair Pair of inputs containing Halo and periodic shift View.
      This pair is returned by the gridHalo function.

      \param PositionIndex Particle position index within the AoSoA.
    */
    PeriodicHalo( std::pair<HaloType, ShiftViewType> pair )
        : _halo( pair.first )
        , _shifts( pair.second )
    {
    }
    ~PeriodicHalo() {}

    auto getHalo() const { return _halo; }

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION void modify_buffer( ViewType &send_buffer,
                                               const int i ) const
    {
        for ( int d = 0; d < 3; ++d )
            get<PositionIndex>( send_buffer( i ), d ) += _shifts( i, d );
    }
};

//---------------------------------------------------------------------------//
/*!
  \brief Create a periodic halo.

  \param local_grid The local grid for creating halo and periodicity.

  \param particles The particle AoSoA, containing positions.

  \param PositionIndex Particle position index within the AoSoA.

  \param min_halo_width Number of halo mesh widths to include for ghosting.

  \param max_export_guess The allocation size for halo export ranks, IDs, and
  periodic shifts.

  \return pair A std::pair containing the Halo and periodic shift array.
*/
template <class LocalGridType, class ParticleContainer,
          std::size_t PositionIndex>
auto createPeriodicHalo( const LocalGridType &local_grid,
                         const ParticleContainer &particles,
                         std::integral_constant<std::size_t, PositionIndex>,
                         const int min_halo_width,
                         const int max_export_guess = 0 )
{
    using view_type =
        Kokkos::View<double **, typename ParticleContainer::device_type>;
    using halo_type = Halo<typename ParticleContainer::device_type>;

    auto positions = slice<PositionIndex>( particles );
    std::pair<halo_type, view_type> pair =
        gridHalo( local_grid, positions, min_halo_width, max_export_guess );

    using phalo_type =
        PeriodicHalo<halo_type, view_type, ParticleContainer, PositionIndex>;
    return phalo_type( pair );
}

//---------------------------------------------------------------------------//
/*!
  \brief gridGather gathers data from one decomposition and ghosts on
  another decomposition, using the bounds and periodicity of a Cajita grid
  to determine which particles should be copied. AoSoA variant.

  \tparam PeriodicHaloType Periodic halo type.

  \tparam ParticleContainer AoSoA type.

  \param periodic_halo The halo and periodicity shift details.

  \param particles The particle AoSoA, containing positions.
*/
template <class PeriodicHaloType, class ParticleContainer>
void gridGather( const PeriodicHaloType &periodic_halo,
                 ParticleContainer &particles )
{
    auto halo = periodic_halo.getHalo();
    particles.resize( halo.numLocal() + halo.numGhost() );

    gather( halo, particles, periodic_halo );
}

// TODO: slice version

} // end namespace Cabana

#endif // end CABANA_PERIODICCOMM_HPP
