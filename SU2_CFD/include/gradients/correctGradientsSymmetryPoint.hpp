/*!
 * @file correctGradientsSymmetryPoint.hpp
 * @brief Per-vertex body of the gradient correction on symmetry planes and Euler walls, shared by the CPU
 *        implementation (correctGradientsSymmetry.hpp) and the CUDA kernel.
 * @note The expression shapes and the loop order are those of the original CPU implementation. Do not
 *       reassociate, hoist or change the types of the expressions in it. The function contains no AD
 *       preaccumulation calls (the original code had none either).
 * @author N. Beishuizen
 * @version 8.6.0 "Harrier"
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

namespace detail {

/*!
 * @brief a.b for the first nDim entries, summed in order starting from zero (as GeometryToolbox::DotProduct).
 */
template <size_t nDim, class A, class B>
SU2_HOST_DEVICE inline su2double symDotProduct(const A& a, const B& b) {
  su2double d(0);
  for (size_t i = 0; i < nDim; ++i) d += a[i] * b[i];
  return d;
}

/*!
 * @brief Correct the gradient on a symmetry plane.
 * @ingroup FvmAlgos
 * @tparam nDim - Number of dimensions.
 * @tparam Matrix - Type for which gradients[iVar] is an object with operator[](iDim) (a row of the gradient).
 * @tparam Scalar - Type of the normal.
 * @param[in] varBegin - Start of the variables.
 * @param[in] varEnd - End of the variables.
 * @param[in] idxVel - Variable index where velocity gradients start (-1 if all variables are scalars).
 * @param[in] n - Normal direction (unit).
 * @param[in, out] gradients - The gradients to be modified.
 */
template <size_t nDim, class Matrix, class Scalar>
SU2_HOST_DEVICE inline void correctGradient(const size_t varBegin, const size_t varEnd, const int idxVel,
                                            const Scalar* n, Matrix&& gradients) {

  /*--- First we correct the part that involves velocities. ---*/

  if (idxVel >= static_cast<int>(varBegin) && idxVel + nDim <= varEnd) {
    /*--- Normal gradient of velocity components and gradient of normal velocity. ---*/
    su2double normalGrad[nDim] = {}, gradNormalVel[nDim] = {};

    for (size_t iDim = 0; iDim < nDim; iDim++) {
      normalGrad[iDim] = symDotProduct<nDim>(gradients[idxVel + iDim], n);

      for (size_t jDim = 0; jDim < nDim; jDim++) {
        gradNormalVel[jDim] += n[iDim] * gradients[idxVel + iDim][jDim];
      }
    }

    /*--- Normal gradient of the normal velocity. ---*/
    const su2double normalGradNormalVel = symDotProduct<nDim>(n, gradNormalVel);

    /*--- Remove the tangential projection (I - n.n^T) of the normal gradients.
     * And the normal projection (n.n^T) of the tangential gradients.
     * dV = dV - (I - n.n^T) dV n.n^T - n.n^T dV (I - n.n^T) ---*/

    for (size_t iDim = 0; iDim < nDim; iDim++) {
      for (size_t jDim = 0; jDim < nDim; jDim++) {
        gradients[idxVel + iDim][jDim] -= normalGrad[iDim] * n[jDim] + n[iDim] * gradNormalVel[jDim];
        gradients[idxVel + iDim][jDim] += 2 * n[iDim] * normalGradNormalVel * n[jDim];
      }
    }
  }

  /*--- Remove the normal component for all scalars (excluding velocities). ---*/

  for (auto iVar = varBegin; iVar < varEnd; iVar++) {
    if (idxVel != -1 && static_cast<int>(iVar) >= idxVel && iVar < idxVel + nDim) continue;

    const su2double normalGrad = symDotProduct<nDim>(n, gradients[iVar]);
    for (size_t iDim = 0; iDim < nDim; iDim++) {
      gradients[iVar][iDim] -= normalGrad * n[iDim];
    }
  }
}

}  // namespace detail
