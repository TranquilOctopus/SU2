/*!
 * @file greenGaussPoint.hpp
 * @brief Per-point body of the Green-Gauss gradient (volume/edge contribution), shared by the CPU
 *        implementation (computeGradientsGreenGauss.hpp) and, in a later step, a CUDA kernel.
 * @note The function is a gather: it only writes the gradient of the point it is given, and it visits the
 *       neighbors in storage order, so the summation order (and therefore the result) is the same on
 *       every backend. Do not reassociate, hoist or change the types of the expressions in it.
 *       Only plain arithmetic is allowed here (no std containers, virtual calls, exceptions or MPI).
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2026, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cstddef>

#include "../../../Common/include/code_config.hpp"

namespace gg {

/*!
 * @brief Raw, read-only view of the geometry needed by the Green-Gauss volume contribution.
 * @note Neighbors and edges share one CSR structure: the outer pointers are those of the point adjacency
 *       (CPoint::Point), the edge indices (CPoint::Edge) have the same sparsity pattern by construction.
 * @note The index types are those of the host containers (CCompressedSparsePatternUL / L, whose width depends
 *       on the build), so views of host data and plain copies of that data on a device are interchangeable.
 */
template <class Scalar, class PointIdx, class EdgeIdx>
struct GeomView {
  const PointIdx* nbrOuter = nullptr;       /*!< @brief Start of each point's row, size nPoint+1. */
  const PointIdx* nbrIdx = nullptr;         /*!< @brief Neighbor point of each CSR entry. */
  const EdgeIdx* edgeIdx = nullptr;         /*!< @brief Edge connecting the point and the neighbor, per CSR entry. */
  const Scalar* normal = nullptr;           /*!< @brief Edge normals, row-major, one row per edge. */
  size_t normalLd = 0;                      /*!< @brief Row length of "normal" (number of dimensions). */
  const Scalar* volume = nullptr;           /*!< @brief Control volume of each point. */
  const Scalar* periodicVolume = nullptr;   /*!< @brief Missing volume of points on periodic boundaries. */
};

/*!
 * @brief Row-major (iPoint, iVar) view of a field, for use as the "field" argument of greenGaussPoint.
 * @note The CPU path passes its existing accessor objects instead; any type with operator(i,j) works.
 */
template <class Scalar>
struct FieldView {
  const Scalar* ptr = nullptr;
  size_t ld = 0; /*!< @brief Row length, i.e. the number of variables stored per point. */

  SU2_HOST_DEVICE const Scalar& operator()(size_t iPoint, size_t iVar) const { return ptr[iPoint * ld + iVar]; }
};

/*!
 * @brief Row-major (iPoint, iVar, iDim) view of a gradient, for use as the "grad" argument of greenGaussPoint.
 * @note The CPU path passes its existing accessor objects instead; any type with operator(i,j,k) works.
 */
template <class Scalar, size_t nDim>
struct GradView {
  Scalar* ptr = nullptr;
  size_t nVar = 0; /*!< @brief Number of variables stored per point. */

  SU2_HOST_DEVICE Scalar& operator()(size_t iPoint, size_t iVar, size_t iDim) const {
    return ptr[(iPoint * nVar + iVar) * nDim + iDim];
  }
};

/*!
 * @brief Preaccumulation policy that does nothing, for code without AD (e.g. device kernels).
 */
struct NoPreacc {
  SU2_HOST_DEVICE static void Start() {}
  template <class T>
  SU2_HOST_DEVICE static void In(T&&) {}
  template <class T>
  SU2_HOST_DEVICE static void InArray(const T&, size_t) {}
  template <class T>
  SU2_HOST_DEVICE static void Out(T&&) {}
  SU2_HOST_DEVICE static void End() {}
};

/*!
 * @brief Evaluate the Green-Gauss volume (edge) contribution to the gradient of one point.
 * @note The expression shapes and the loop order are exactly those of the original CPU implementation:
 *       weight * (a + b), then "+= flux * area[iDim]" over the neighbors in storage order.
 * @tparam nDim - Number of dimensions (to allow unrolling).
 * @tparam Scalar - su2double of the build (double, or an AD type).
 * @tparam Preacc - Policy providing Start(), In(x), InArray(ptr, n), Out(x), End(), see NoPreacc and, for the
 *         host, AdPreacc (computeGradientsGreenGauss.hpp). The calls sit at the same places as the original
 *         AD::StartPreacc/SetPreaccIn/SetPreaccOut/EndPreacc calls.
 * @tparam Field - Type with operator()(iPoint, iVar) returning the field value.
 * @tparam Grad - Type with operator()(iPoint, iVar, iDim) returning a reference to the gradient entry.
 * @param[in] iPoint - Point (must be a domain point).
 * @param[in] g - Geometry view.
 * @param[in] field - The field.
 * @param[in] varBegin - First variable.
 * @param[in] varEnd - End of the variable range (exclusive).
 * @param[in,out] grad - Gradient, the entries of iPoint are overwritten.
 */
template <size_t nDim, class Scalar, class Preacc, class GeomView, class Field, class Grad>
SU2_HOST_DEVICE inline void greenGaussPoint(size_t iPoint, const GeomView& g, const Field& field,
                                            size_t varBegin, size_t varEnd, Grad& grad) {
  Preacc::Start();
  Preacc::In(g.volume[iPoint]);
  Preacc::In(g.periodicVolume[iPoint]);

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar) Preacc::In(field(iPoint, iVar));

  /*--- Clear the gradient. --*/

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
    for (size_t iDim = 0; iDim < nDim; ++iDim) grad(iPoint, iVar, iDim) = 0.0;

  /*--- Handle averaging and division by volume in one constant. ---*/

  Scalar halfOnVol = 0.5 / (g.volume[iPoint] + g.periodicVolume[iPoint]);

  /*--- Add a contribution due to each neighbor. ---*/

  const size_t rowBegin = g.nbrOuter[iPoint];
  const size_t nNeigh = g.nbrOuter[iPoint + 1] - rowBegin;

  for (size_t iNeigh = 0; iNeigh < nNeigh; ++iNeigh) {
    size_t iEdge = g.edgeIdx[rowBegin + iNeigh];
    size_t jPoint = g.nbrIdx[rowBegin + iNeigh];

    /*--- Determine if edge points inwards or outwards of iPoint.
     *    If inwards we need to flip the area vector. ---*/

    Scalar dir = (iPoint < jPoint) ? 1.0 : -1.0;
    Scalar weight = dir * halfOnVol;

    const Scalar* area = g.normal + iEdge * g.normalLd;
    Preacc::InArray(area, nDim);

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
      Preacc::In(field(jPoint, iVar));
      Scalar flux = weight * (field(iPoint, iVar) + field(jPoint, iVar));

      for (size_t iDim = 0; iDim < nDim; ++iDim) grad(iPoint, iVar, iDim) += flux * area[iDim];
    }
  }

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
    for (size_t iDim = 0; iDim < nDim; ++iDim) Preacc::Out(grad(iPoint, iVar, iDim));

  Preacc::End();
}

}  // namespace gg
