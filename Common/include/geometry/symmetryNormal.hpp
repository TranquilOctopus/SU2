/*!
 * @file symmetryNormal.hpp
 * @brief Unit normal used to correct the gradients at a vertex of a symmetry or Euler wall marker.
 * @note Shared by the CPU gradient correction and by the upload of the normals to the device (CGeometryDevice),
 *       so that both use the same arithmetic.
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

#include "CGeometry.hpp"
#include "../toolboxes/geometry_toolbox.hpp"

/*!
 * @brief Get the unit normal of a vertex of a symmetry or Euler wall marker. This may be the original normal of the
 *        vertex or a modified normal if there are intersecting symmetries (see
 *        CGeometry::ComputeModifiedSymmetryNormals).
 * @tparam nDim - Number of dimensions.
 * @param[in] geometry - Geometric grid properties.
 * @param[in] iMarker - Marker.
 * @param[in] iVertex - Vertex of the marker.
 * @param[out] unitNormal - Array of nDim entries.
 */
template <size_t nDim>
inline void GetSymmetryUnitNormal(const CGeometry& geometry, size_t iMarker, size_t iVertex, su2double* unitNormal) {
  for (size_t iDim = 0; iDim < nDim; iDim++) unitNormal[iDim] = 0.0;

  const auto it = geometry.symmetryNormals[iMarker].find(iVertex);

  if (it != geometry.symmetryNormals[iMarker].end()) {
    for (auto iDim = 0u; iDim < nDim; iDim++) unitNormal[iDim] = it->second[iDim];
  } else {
    geometry.vertex[iMarker][iVertex]->GetNormal(unitNormal);
    const su2double area = GeometryToolbox::Norm(nDim, unitNormal);
    for (auto iDim = 0u; iDim < nDim; iDim++) unitNormal[iDim] /= area;
  }
}
