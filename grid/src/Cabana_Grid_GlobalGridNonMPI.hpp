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
  \file Cabana_Grid_GlobalGridNonMPI.hpp
  \brief Global grid without MPI
*/
#ifndef CABANA_GRID_GLOBALGRIDNONMPI_HPP
#define CABANA_GRID_GLOBALGRIDNONMPI_HPP

#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_Types.hpp>

#include <array>
#include <memory>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
/*!
  \brief Global logical grid without MPI.
  \tparam MeshType Mesh type (uniform, non-uniform, sparse)
*/
template <class MeshType>
class GlobalGridBase
{
  public:
    //! Mesh type.
    using mesh_type = MeshType;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    /*!
     \brief Constructor.
     \param comm The communicator over which to define the grid.
     \param global_mesh The global mesh data.
     \param periodic Whether each logical dimension is periodic.
     \param partitioner The grid partitioner.
    */
    GlobalGridBase( const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
                    const std::array<bool, num_space_dim>& periodic )
        : _global_mesh( global_mesh )
        , _periodic( periodic )
    {
        // All global cells owned.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _owned_num_cell[d] = _global_mesh->globalNumCell( d );

        // Extract the periodicity of the boundary as integers.
        std::array<int, num_space_dim> periodic_dims;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            periodic_dims[d] = _periodic[d];

        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            _ranks_per_dim[d] = 1;
            _cart_rank[d] = 0;
            _global_cell_offset[d] = 0;
            _boundary_lo[d] = true;
            _boundary_hi[d] = true;
        }
    }

    // Destructor.
    ~GlobalGridBase(){};

    //! \brief Get the global mesh data.
    const GlobalMesh<MeshType>& globalMesh() const { return *_global_mesh; }

    //! \brief Get whether a given dimension is periodic.
    bool isPeriodic( const int dim ) const { return _periodic[dim]; }

    //! \brief Determine if this block is on a low boundary in this dimension.
    //! \param dim Spatial dimension.
    bool onLowBoundary( const int dim ) const { return _boundary_lo[dim]; }

    //! \brief Determine if this block is on a high boundary in this dimension.
    //! \param dim Spatial dimension.
    bool onHighBoundary( const int dim ) const { return _boundary_hi[dim]; }

    //! \brief Get the number of blocks in each dimension in the global mesh.
    //! \param dim Spatial dimension.
    int dimNumBlock( const int dim ) const { return _ranks_per_dim[dim]; }

    //! \brief Get the total number of blocks.
    int totalNumBlock() const { return 1; }

    //! \brief Get the id of this block in a given dimension.
    //! \param dim Spatial dimension.
    int dimBlockId( const int ) const { return 0; }

    //! \brief Get the id of this block.
    int blockId() const { return 0; }

    /*!
      \brief Get the MPI rank of a block with the given indices. If the rank is
      out of bounds and the boundary is not periodic, return -1 to indicate an
      invalid rank.

      \param ijk %Array of block indices.
    */
    int blockRank( const std::array<int, num_space_dim>& ijk ) const
    {
        // Check for invalid indices. An index is invalid if it is out of bounds
        // and the dimension is not periodic. An out of bound index in a
        // periodic dimension is valid because it will wrap around to a valid
        // index.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            if ( !_periodic[d] &&
                 ( ijk[d] < 0 || _ranks_per_dim[d] <= ijk[d] ) )
                return -1;

        return 0;
    }

    /*!
      \brief Get the MPI rank of a block with the given indices. If the rank is
      out of bounds and the boundary is not periodic, return -1 to indicate an
      invalid rank.

      \param i,j,k Block index.
    */
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> blockRank( const int i, const int j,
                                               const int k ) const
    {
        std::array<int, 3> cr = { i, j, k };
        return blockRank( cr );
    }

    /*!
      \brief Get the MPI rank of a block with the given indices. If the rank is
      out of bounds and the boundary is not periodic, return -1 to indicate an
      invalid rank.

      \param i,j Block index.
    */
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, int> blockRank( const int i, const int j ) const
    {
        std::array<int, 2> cr = { i, j };
        return blockRank( cr );
    }

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumEntity( Cell, const int dim ) const
    {
        return _global_mesh->globalNumCell( dim );
    }

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumEntity( Node, const int dim ) const
    {
        // If this dimension is periodic that last node in the dimension is
        // repeated across the periodic boundary.
        if ( _periodic[dim] )
            return globalNumEntity( Cell(), dim );
        else
            return globalNumEntity( Cell(), dim ) + 1;
    }

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumEntity( Face<Dim::I>, const int dim ) const
    {
        return ( Dim::I == dim ) ? globalNumEntity( Node(), dim )
                                 : globalNumEntity( Cell(), dim );
    }

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumEntity( Face<Dim::J>, const int dim ) const
    {
        return ( Dim::J == dim ) ? globalNumEntity( Node(), dim )
                                 : globalNumEntity( Cell(), dim );
    }

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> globalNumEntity( Face<Dim::K>,
                                                     const int dim ) const
    {
        return ( Dim::K == dim ) ? globalNumEntity( Node(), dim )
                                 : globalNumEntity( Cell(), dim );
    }

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> globalNumEntity( Edge<Dim::I>,
                                                     const int dim ) const
    {
        return ( Dim::I == dim ) ? globalNumEntity( Cell(), dim )
                                 : globalNumEntity( Node(), dim );
    }

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> globalNumEntity( Edge<Dim::J>,
                                                     const int dim ) const
    {
        return ( Dim::J == dim ) ? globalNumEntity( Cell(), dim )
                                 : globalNumEntity( Node(), dim );
    }

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> globalNumEntity( Edge<Dim::K>,
                                                     const int dim ) const
    {
        return ( Dim::K == dim ) ? globalNumEntity( Cell(), dim )
                                 : globalNumEntity( Node(), dim );
    }

    //! \brief Get the owned number of cells in a given dimension of this block.
    //! \param dim Spatial dimension.
    int ownedNumCell( const int dim ) const { return _owned_num_cell[dim]; }

    //! \brief Get the global offset in a given dimension. This is where our
    //! block starts in the global indexing scheme.
    //! \param dim Spatial dimension.
    int globalOffset( const int dim ) const { return _global_cell_offset[dim]; }

    //! \brief Set number of cells and offset of local part of the grid. Make
    //! sure these are consistent across all ranks.
    //! \param num_cell New number of owned cells for all dimensions.
    //! \param offset New global offset for all dimensions.
    void setNumCellAndOffset( const std::array<int, num_space_dim>& num_cell,
                              const std::array<int, num_space_dim>& offset );

  private:
    std::shared_ptr<GlobalMesh<MeshType>> _global_mesh;
    std::array<bool, num_space_dim> _periodic;
    std::array<int, num_space_dim> _ranks_per_dim;
    std::array<int, num_space_dim> _cart_rank;
    std::array<int, num_space_dim> _owned_num_cell;
    std::array<int, num_space_dim> _global_cell_offset;
    std::array<bool, num_space_dim> _boundary_lo;
    std::array<bool, num_space_dim> _boundary_hi;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a global grid.
  \param global_mesh The global mesh data.
  \param periodic Whether each logical dimension is periodic.
*/
template <class MeshType>
std::shared_ptr<GlobalGridBase<MeshType>> createGlobalGridBase(
    const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
    const std::array<bool, MeshType::num_space_dim>& periodic )
{
    return std::make_shared<GlobalGridBase<MeshType>>( global_mesh, periodic );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana
//---------------------------------------------------------------------------//

#endif // end CABANA_GRID_GLOBALGRID_HPP
