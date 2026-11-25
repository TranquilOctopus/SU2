/*!
 * @file CGeometryDevice.hpp
 * @brief Read-only snapshot of the mesh geometry on the GPU.
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

#include "../code_config.hpp"

/*--- Only exists in builds with CUDA that are not AD builds (see code_config.hpp), the header is empty otherwise. ---*/
#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../containers/CDeviceMirror.hpp"
#include "../toolboxes/graph_toolbox.hpp"
#include "CDeviceHaloExchange.hpp"
#include "GeomView.hpp"

class CGeometry;
class CConfig;

/*!
 * @class CGeometryDevice
 * @brief Copy of the (MESH_0) geometry arrays needed by the GPU kernels, uploaded once and read-only afterwards.
 * @details Holds the point adjacency (CSR) with the edge index of each entry, edge normals, control volumes,
 * periodic volumes, domain flags, and for each marker its type, vertex points and vertex normals (plus, for symmetry
 * and Euler wall markers, the unit normals used by the gradient correction). Index and
 * value types are exactly those of the host containers. The snapshot does not follow later changes of the
 * host geometry (e.g. mesh motion), it is built from the geometry as it is when the object is created.
 * Objects are created by CGeometry::GetDevice, which is not thread safe: call it outside parallel regions.
 * Out of memory and other CUDA errors end the run with SU2_MPI::Error.
 */
class CGeometryDevice {
 public:
  using PointIdx = CCompressedSparsePatternUL::IndexType;  /*!< @brief Type of the point indices of the adjacency. */
  using EdgeIdx = CCompressedSparsePatternL::IndexType;    /*!< @brief Type of the edge indices of the adjacency. */
  using NodeIdx = unsigned long;                           /*!< @brief Type of the point index of a vertex. */
  using View = gg::GeomView<su2double, PointIdx, EdgeIdx>; /*!< @brief Same type as the view of the host data. */

  /*! @brief Raw device pointers of the data of one marker. */
  struct MarkerView {
    unsigned short kindBC = 0;         /*!< @brief Type of boundary condition (from the config). */
    size_t nVertex = 0;                /*!< @brief Number of vertices. */
    const NodeIdx* node = nullptr;     /*!< @brief Point of each vertex. */
    const su2double* normal = nullptr; /*!< @brief Vertex normals, row-major, one row (of length nDim) per vertex. */
    /*! @brief Unit normals for the gradient correction (see GetSymmetryUnitNormal), same layout as normal. Only
     *         for symmetry and Euler wall markers, nullptr for the others. */
    const su2double* unitNormal = nullptr;
  };

 private:
  /*! @brief Staging buffers (owned by this class) and mirrors of one marker. */
  struct Marker {
    unsigned short kindBC = 0;
    std::vector<NodeIdx> hostNode;
    std::vector<su2double> hostNormal;
    std::vector<su2double> hostUnitNormal;
    CDeviceMirror<NodeIdx> node;
    CDeviceMirror<su2double> normal;
    CDeviceMirror<su2double> unitNormal;
  };

  size_t nPoint = 0, nPointDomain = 0, nEdge = 0, nDim = 0;
  size_t bytes = 0; /*!< @brief Total size of the device arrays. */

  /*--- Arrays that are contiguous on the host are mirrored in place, the others go through staging buffers. ---*/
  CDeviceMirror<PointIdx> nbrOuter, nbrIdx;
  CDeviceMirror<EdgeIdx> edgeIdx;
  CDeviceMirror<su2double> edgeNormal, volume, periodicVolume;
  std::vector<uint8_t> hostDomain;
  CDeviceMirror<uint8_t> domain;
  std::vector<Marker> markers;
  std::unique_ptr<CDeviceHaloExchange> halo;

  void Upload(const CGeometry& geometry, const CConfig& config);

  template <class T>
  void Send(CDeviceMirror<T>& mirror, const T* hostPtr, size_t n, const std::string& name);

 public:
  /*!
   * @brief Build the snapshot.
   * @param[in] geometry - Finest grid, with edges, vertices and control volumes already set up.
   * @param[in] config - Configuration, for the type of each marker.
   */
  CGeometryDevice(const CGeometry& geometry, const CConfig& config);

  CGeometryDevice(const CGeometryDevice&) = delete;
  CGeometryDevice& operator=(const CGeometryDevice&) = delete;

  size_t GetnPoint() const { return nPoint; }
  size_t GetnPointDomain() const { return nPointDomain; }
  size_t GetnEdge() const { return nEdge; }
  size_t GetnDim() const { return nDim; }
  size_t GetnMarker() const { return markers.size(); }

  /*! @brief Total size in bytes of the device arrays. */
  size_t GetBytes() const { return bytes; }

  /*! @brief View of device pointers for gg::greenGaussPoint, same type as the host view. */
  View GetGeomView() const;

  /*! @brief Device array with one flag per point (1 for domain points, 0 for halos). */
  const uint8_t* GetDomain() const { return domain.DeviceRead(); }

  /*! @brief Device pointers and sizes of one marker. */
  MarkerView GetMarker(size_t iMarker) const;

  /*! @brief MPI halo exchange of device arrays (non-const on purpose: it owns staging buffers, not geometry data). */
  CDeviceHaloExchange& GetHalo() const { return *halo; }
};

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
