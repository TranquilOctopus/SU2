/*!
 * @file GeomView.hpp
 * @brief Raw read-only view of the mesh adjacency and dual grid, shared by the CPU and the device.
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

namespace gg {

/*!
 * @brief Raw, read-only view of the geometry needed by the Green-Gauss volume contribution and the least-squares gradient.
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
  const Scalar* coord = nullptr;            /*!< @brief Point coordinates, row-major, one row per point (least squares). */
  size_t coordLd = 0;                       /*!< @brief Row length of "coord" (number of dimensions). */
};

}  // namespace gg
