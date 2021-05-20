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

#ifndef CABANA_XYZPARTICLEWRITER_HPP
#define CABANA_XYZPARTICLEWRITER_HPP

#include <Cabana_AoSoA.hpp>

#include <array>
#include <fstream>
#include <iostream>

namespace Cabana
{

template <class StreamType, class CoordSliceType>
void writeXyzPositions( StreamType& stream, const CoordSliceType& coords )
{
    for ( std::size_t p = 0; p < coords.size(); p++ )
    {
        for ( int d = 0; d < 3; d++ )
            stream << coords( p, d ) << " ";
        stream << std::endl;
    }
}

template <class CoordSliceType>
void writeXyzTimeStep( std::string filename, const int time_step_index,
                       const double time, const CoordSliceType& coords )
{
    std::ofstream stream( filename );

    stream << coords.size() << std::endl;

    // Comment line
    stream << time_step_index << " " << time << std::endl;
    writeXyzPositions( stream, coords );
}

// Write a time step.
template <class CoordSliceType>
void writeXyzTimeStep( std::string filename, const int time_step_index,
                       const double time, std::array<double, 3>& global_low,
                       std::array<double, 3>& global_high,
                       const CoordSliceType& coords )
{
    std::ofstream stream( filename );

    stream << coords.size() << std::endl;

    // Comment line
    stream << "Lattice=\"" << global_low[0] << " " << global_low[1] << " "
           << global_low[2] << " " << global_high[0] << " " << global_high[1]
           << " " << global_high[2] << "\" Index=\"" << time_step_index
           << "\" Time=\"" << time << std::endl;
    writeXyzPositions( stream, coords );
}

template <class CoordSliceType>
void read_line( CoordSliceType& positions, const int p, std::string line,
                std::string delim = " " )
{
    int d = 0;
    int begin = 0;
    int end = line.find( delim );
    while ( end != -1 )
    {
        positions( p, d ) = std::stod( line.substr( begin, end - begin ) );
        begin = end + delim.size();
        end = line.find( delim, begin );
        d++;
    }
}

template <class Particles>
void readXyzTimeStep( std::string filename, Particles& particles )
{
    std::ifstream stream( filename );
    std::string line;

    // First line: number of particles
    std::getline( stream, line );
    std::size_t num_particle = std::stoi( line );
    particles.resize( num_particle );
    auto positions = Cabana::slice<0>( particles );

    // Second line: comment
    std::getline( stream, line );

    // Particle data.
    for ( std::size_t p = 0; p < num_particle; p++ )
    {
        std::getline( stream, line );
        read_line( positions, p, line );
    }
}

} // namespace Cabana
#endif
