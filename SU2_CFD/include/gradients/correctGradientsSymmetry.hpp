/*!
 * \file correctGradientsSymmetry.hpp
 * \brief Implements the symmetry boundary conditions for the gradient computations.
 * \author N. Beishuizen
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

#include <vector>
#include <algorithm>

#include "../../../Common/include/parallelization/omp_structure.hpp"
#include "../../../Common/include/geometry/symmetryNormal.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"
#include "correctGradientsSymmetryPoint.hpp"

/*!
 * \brief Correct gradients on symmetry and Euler (slip) markers to respect the conditions:
 *        1. n.grad(phi) = 0
 *        2. n.grad(v.t) = 0
 *        3. t.grad(v.n) = 0
 * \note See Blazek eq. 8.40.
 * \ingroup FvmAlgos
 * \param[in] geometry - Geometric grid properties.
 * \param[in] config - Configuration of the problem, used to identify types of boundaries.
 * \param[in] varBegin - Index of first variable for which to compute the gradient.
 * \param[in] varEnd - Index of last variable for which to compute the gradient.
 * \param[in] idxVel - Index to velocity, -1 if no velocity is present in the solver.
 * \param[in,out] gradient - Generic object implementing operator (iPoint, iVar, iDim).
 */
template <size_t nDim, class GradientType>
void correctGradientsSymmetry(CGeometry& geometry, const CConfig& config, const size_t varBegin,
                              const size_t varEnd, const int idxVel, GradientType& gradient) {

  /*--- Check how many symmetry planes there are. ---*/
  std::vector<unsigned short> symMarkers;
  for (auto iMarker = 0u; iMarker < geometry.GetnMarker(); ++iMarker) {
    if (config.GetMarker_All_KindBC(iMarker) == SYMMETRY_PLANE ||
        config.GetMarker_All_KindBC(iMarker) == EULER_WALL) {
      symMarkers.push_back(iMarker);
    }
  }

  for (const auto iMarker : symMarkers) {
    SU2_OMP_FOR_STAT(32)
    for (size_t iVertex = 0; iVertex < geometry.GetnVertex(iMarker); ++iVertex) {

      const auto iPoint = geometry.vertex[iMarker][iVertex]->GetNode();

      /*--- Get the normal of the current symmetry. This may be the original normal of the vertex
       * or a modified normal if there are intersecting symmetries. ---*/

      su2double unitNormal[nDim] = {};
      GetSymmetryUnitNormal<nDim>(geometry, iMarker, iVertex, unitNormal);

      detail::correctGradient<nDim>(varBegin, varEnd, idxVel, unitNormal, gradient[iPoint]);

    } // loop over vertices
    END_SU2_OMP_FOR
  } // markers

}

