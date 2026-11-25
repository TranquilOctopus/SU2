/*!
 * \file leastSquaresPoint.hpp
 * \brief Per-point Least-Squares gradient functions, shared by the CPU and the GPU (CUDA) implementations.
 * \note The expression shapes, the neighbor order and the 3D "Rmatrix(iPoint,2,1)" detail are exactly those of the
 *       original CPU implementation (computeGradientsLeastSquares.hpp). Automatic differentiation calls are made
 *       through a preaccumulation policy (see gg::NoPreacc and detail::AdPreacc) at the places of the original ones.
 * \version 8.6.0 "Harrier"
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

#include <cmath>
#include <cstddef>
#include <limits>

#include "../../../Common/include/code_config.hpp"
#include "../../../Common/include/geometry/GeomView.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"
#include "greenGaussPoint.hpp"

namespace ls {

/*!
 * @brief Prepare Smatrix for 2D.
 * @ingroup FvmAlgos
 */
SU2_HOST_DEVICE FORCEINLINE void computeSmatrix(su2double r11, su2double r12, su2double r13, su2double r22,
                                           su2double r23, su2double r33, su2double detR2, su2double Smatrix[][2]) {
  Smatrix[0][0] = (r12 * r12 + r22 * r22) / detR2;
  Smatrix[0][1] = -r11 * r12 / detR2;
  Smatrix[1][1] = r11 * r11 / detR2;
}

/*!
 * @brief Prepare Smatrix for 3D.
 * @ingroup FvmAlgos
 */
SU2_HOST_DEVICE FORCEINLINE void computeSmatrix(su2double r11, su2double r12, su2double r13, su2double r22,
                                           su2double r23, su2double r33, su2double detR2, su2double Smatrix[][3]) {
  su2double z11 = r22 * r33;
  su2double z12 = -r12 * r33;
  su2double z13 = r12 * r23 - r13 * r22;
  su2double z22 = r11 * r33;
  su2double z23 = -r11 * r23;
  su2double z33 = r11 * r22;

  Smatrix[0][0] = (z11 * z11 + z12 * z12 + z13 * z13) / detR2;
  Smatrix[0][1] = (z12 * z22 + z13 * z23) / detR2;
  Smatrix[0][2] = (z13 * z33) / detR2;
  Smatrix[1][1] = (z22 * z22 + z23 * z23) / detR2;
  Smatrix[1][2] = (z23 * z33) / detR2;
  Smatrix[2][2] = (z33 * z33) / detR2;
}

/*!
 * @brief Small dense (iDim, jDim) matrix of one point with the interface of the Rmatrix container (the point index
 *        is ignored). The GPU kernels use it to keep the R matrix of the thread in registers, it is not stored.
 */
template <size_t nDim>
struct LocalRMatrix {
  su2double m[nDim][nDim];

  SU2_HOST_DEVICE su2double& operator()(size_t, size_t iDim, size_t jDim) { return m[iDim][jDim]; }
  SU2_HOST_DEVICE const su2double& operator()(size_t, size_t iDim, size_t jDim) const { return m[iDim][jDim]; }
};

/*!
 * @brief Solve the least-squares problem for one point.
 * @ingroup FvmAlgos
 * @note The template "nDim" allows unrolling. "periodic" selects the version that runs after the periodic
 *       communications (it starts and ends its own preaccumulation region, the gradient is in/out), the other
 *       version is the end of the preaccumulation region that was started by leastSquaresAccumulate.
 * @tparam Preacc - Preaccumulation policy (see gg::NoPreacc).
 * @param[in] iPoint - Point.
 * @param[in] varBegin - First variable.
 * @param[in] varEnd - End of the variable range (exclusive).
 * @param[in] Rmatrix - Object implementing operator (iPoint, iDim, jDim).
 * @param[in,out] gradient - Object implementing operator (iPoint, iVar, iDim), holds the right-hand side on entry.
 */
template <size_t nDim, bool periodic, class Preacc, class GradientType, class RMatrixType>
SU2_HOST_DEVICE FORCEINLINE void solveLeastSquares(size_t iPoint, size_t varBegin, size_t varEnd,
                                              const RMatrixType& Rmatrix, GradientType& gradient) {
  constexpr passivedouble epsilon = std::numeric_limits<passivedouble>::epsilon();
  const auto eps = pow(epsilon, 2);

  /*--- Entries of upper triangular matrix R. ---*/

  if (periodic) {
    Preacc::StartUnguarded();
    Preacc::In(Rmatrix(iPoint, 0, 0));
    Preacc::In(Rmatrix(iPoint, 0, 1));
    Preacc::In(Rmatrix(iPoint, 1, 1));
  }

  su2double r11 = Rmatrix(iPoint, 0, 0);
  su2double r12 = Rmatrix(iPoint, 0, 1);
  su2double r22 = Rmatrix(iPoint, 1, 1);
  su2double r13 = 0.0, r23 = 0.0, r33 = 1.0;

  r11 = sqrt(max(r11, eps));
  r12 /= r11;
  r22 = sqrt(max(r22 - r12 * r12, eps));

  if (nDim == 3) {
    if (periodic) {
      Preacc::In(Rmatrix(iPoint, 0, 2));
      Preacc::In(Rmatrix(iPoint, 1, 2));
      Preacc::In(Rmatrix(iPoint, 2, 1));
      Preacc::In(Rmatrix(iPoint, 2, 2));
    }

    r13 = Rmatrix(iPoint, 0, 2);
    r33 = Rmatrix(iPoint, 2, 2);
    const auto r23_a = Rmatrix(iPoint, 1, 2);
    const auto r23_b = Rmatrix(iPoint, 2, 1);

    r13 /= r11;
    r23 = r23_a / r22 - r23_b * r12 / (r11 * r22);
    r33 = sqrt(max(r33 - r23 * r23 - r13 * r13, eps));
  }

  /*--- Compute determinant ---*/

  const su2double detR2 = pow(r11 * r22 * r33, 2);

  /*--- S matrix := inv(R)*traspose(inv(R)) ---*/

  su2double Smatrix[nDim][nDim] = {{0.0}};

  /*--- Detect singular matrix ---*/

  if (detR2 > eps) {
    computeSmatrix(r11, r12, r13, r22, r23, r33, detR2, Smatrix);
  }

  if (periodic) {
    /*--- Stop preacc here as gradient is in/out. ---*/
    for (size_t iDim = 0; iDim < nDim; ++iDim)
      for (size_t jDim = iDim; jDim < nDim; ++jDim) Preacc::Out(Smatrix[iDim][jDim]);
    Preacc::End();
  }

  /*--- Computation of the gradient: S*c ---*/

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
    su2double Cvector[nDim] = {0.0};

    for (size_t iDim = 0; iDim < nDim; ++iDim)
      for (size_t jDim = 0; jDim < nDim; ++jDim)
        Cvector[iDim] += Smatrix[min(iDim, jDim)][max(iDim, jDim)] * gradient(iPoint, iVar, jDim);

    for (size_t iDim = 0; iDim < nDim; ++iDim) gradient(iPoint, iVar, iDim) = Cvector[iDim];
  }

  if (!periodic) {
    /*--- Stop preacc here instead as gradient is only out. ---*/
    for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
      for (size_t iDim = 0; iDim < nDim; ++iDim) Preacc::Out(gradient(iPoint, iVar, iDim));
    Preacc::End();
  }
}

/*!
 * @brief Clear and accumulate the R matrix and the right-hand side of the gradient of one point over its neighbors
 *        (inverse-distance weighted or unweighted), in storage order of the neighbors. Starts a preaccumulation
 *        region that is ended by solveLeastSquares<nDim, false> or by the caller (periodic case).
 * @tparam Preacc - Preaccumulation policy (see gg::NoPreacc).
 * @param[in] iPoint - Point (must be a domain point).
 * @param[in] g - Geometry view with the neighbors (nbrOuter, nbrIdx) and the coordinates (coord, coordLd).
 * @param[in] weighted - Use inverse-distance weights.
 * @param[in] field - Object implementing operator (iPoint, iVar).
 * @param[in] varBegin - First variable.
 * @param[in] varEnd - End of the variable range (exclusive).
 * @param[in,out] gradient - Object implementing operator (iPoint, iVar, iDim), the entries of iPoint are overwritten.
 * @param[out] Rmatrix - Object implementing operator (iPoint, iDim, jDim), the entries of iPoint are overwritten.
 */
template <size_t nDim, class Preacc, class GeomView, class FieldType, class GradientType, class RMatrixType>
SU2_HOST_DEVICE FORCEINLINE void leastSquaresAccumulate(size_t iPoint, const GeomView& g, bool weighted,
                                                   const FieldType& field, size_t varBegin, size_t varEnd,
                                                   GradientType& gradient, RMatrixType& Rmatrix) {
  const su2double* coord_i = g.coord + iPoint * g.coordLd;

  Preacc::Start();
  Preacc::InArray(coord_i, nDim);

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar) Preacc::In(field(iPoint, iVar));

  /*--- Clear gradient and Rmatrix. ---*/

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
    for (size_t iDim = 0; iDim < nDim; ++iDim) gradient(iPoint, iVar, iDim) = 0.0;

  for (size_t iDim = 0; iDim < nDim; ++iDim)
    for (size_t jDim = 0; jDim < nDim; ++jDim) Rmatrix(iPoint, iDim, jDim) = 0.0;

  const size_t rowBegin = g.nbrOuter[iPoint];
  const size_t nNeigh = g.nbrOuter[iPoint + 1] - rowBegin;

  for (size_t iNeigh = 0; iNeigh < nNeigh; ++iNeigh) {
    const size_t jPoint = g.nbrIdx[rowBegin + iNeigh];
    const su2double* coord_j = g.coord + jPoint * g.coordLd;
    Preacc::InArray(coord_j, nDim);

    /*--- Distance vector from iPoint to jPoint ---*/

    su2double dist_ij[nDim] = {0.0};
    GeometryToolbox::Distance(nDim, coord_j, coord_i, dist_ij);

    /*--- Compute inverse weight, default 1 (unweighted). ---*/

    su2double weight = 1.0;
    if (weighted) weight = GeometryToolbox::SquaredNorm(nDim, dist_ij);

    /*--- Summations for entries of upper triangular matrix R. ---*/

    if (weight > 0.0) {
      weight = 1.0 / weight;

      for (size_t iDim = 0; iDim < nDim; ++iDim)
        for (size_t jDim = iDim; jDim < nDim; ++jDim)
          Rmatrix(iPoint, iDim, jDim) += dist_ij[iDim] * dist_ij[jDim] * weight;

      if (nDim == 3) Rmatrix(iPoint, 2, 1) += dist_ij[0] * dist_ij[nDim - 1] * weight;

      /*--- Entries of c:= transpose(A)*b ---*/

      for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
        Preacc::In(field(jPoint, iVar));

        su2double delta_ij = weight * (field(jPoint, iVar) - field(iPoint, iVar));

        for (size_t iDim = 0; iDim < nDim; ++iDim) gradient(iPoint, iVar, iDim) += dist_ij[iDim] * delta_ij;
      }
    }
  }
}

/*!
 * @brief The complete least-squares gradient of one point when no periodic communication is needed: accumulate,
 *        solve, and end the preaccumulation region.
 * @note See leastSquaresAccumulate for the arguments.
 */
template <size_t nDim, class Preacc, class GeomView, class FieldType, class GradientType, class RMatrixType>
SU2_HOST_DEVICE FORCEINLINE void leastSquaresPoint(size_t iPoint, const GeomView& g, bool weighted,
                                              const FieldType& field, size_t varBegin, size_t varEnd,
                                              GradientType& gradient, RMatrixType& Rmatrix) {
  leastSquaresAccumulate<nDim, Preacc>(iPoint, g, weighted, field, varBegin, varEnd, gradient, Rmatrix);
  solveLeastSquares<nDim, false, Preacc>(iPoint, varBegin, varEnd, Rmatrix, gradient);
}

}  // namespace ls
