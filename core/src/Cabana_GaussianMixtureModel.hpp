/****************************************************************************
 * Copyright (c) 2023 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_GaussianMixtureModel.hpp
  \brief Creation of a Gaussian Mixture Model
*/
#ifndef CABANA_GMM_HPP
#define CABANA_GMM_HPP

#include <typeinfo>

// Things to describe a Gaussian
enum GaussianFields {
    Weight,
    // Can we make only the relevant fields vissible based on <dims>?
    MuPar, MuPer,
    Cparpar, Cparper,
    Cperpar, Cperper,
    MuX, MuY, MuZ,
    Cxx, Cxy, Cxz,
    Cyx, Cyy, Cyz,
    Czx, Czy, Czz,
    n_gaussian_param
};


#include <impl/Cabana_GaussianMixtureModel.hpp>

namespace Cabana {

template <typename GaussianType, typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
void reconstructGMM(GaussianType& gaussians, const double eps, const int seed, CellSliceType const& cell, WeightSliceType const& weight, VelocitySliceType const& vx) {
	GMMImpl<1>::implReconstructGMM(gaussians, eps, seed, cell, weight, vx, vx, vx);
}

template <typename GaussianType, typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
void reconstructGMM(GaussianType& gaussians, const double eps, const int seed, CellSliceType const& cell, WeightSliceType const& weight, VelocitySliceType const& vpar, VelocitySliceType const& vper) {
	GMMImpl<2>::implReconstructGMM(gaussians, eps, seed, cell, weight, vpar, vper, vper);
}

template <typename GaussianType, typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
void reconstructGMM(GaussianType& gaussians, const double eps, const int seed, CellSliceType const& cell, WeightSliceType const& weight, VelocitySliceType const& vx, VelocitySliceType const& vy, VelocitySliceType const& vz) {
	GMMImpl<3>::implReconstructGMM(gaussians, eps, seed, cell, weight, vx, vy, vz);
}

} // end namespace Cabana

#endif // end CABANA_GMM_HPP
