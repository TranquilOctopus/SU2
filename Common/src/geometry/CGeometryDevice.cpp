/*!
 * @file CGeometryDevice.cpp
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

#include "../../include/geometry/CGeometryDevice.hpp"

#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <exception>
#include <iomanip>
#include <iostream>

#include "../../include/geometry/CGeometry.hpp"

template <class T>
void CGeometryDevice::Send(CDeviceMirror<T>& mirror, const T* hostPtr, size_t n, const std::string& name) {
  /*--- The mirror only reads the host array (nothing is ever downloaded), hence the cast. ---*/
  mirror.Attach(const_cast<T*>(hostPtr), n, "CGeometryDevice::" + name);
  mirror.ToDevice();
  bytes += n * sizeof(T);
}

void CGeometryDevice::Upload(const CGeometry& geometry, const CConfig& config) {
  const auto* nodes = geometry.nodes;
  if (nodes == nullptr || geometry.edges == nullptr) {
    SU2_MPI::Error("The geometry needs its points and edges before a device copy can be made.", CURRENT_FUNCTION);
  }

  const auto& points = nodes->GetPoints();
  const auto& edgePattern = nodes->GetEdges();
  if (points.empty() || edgePattern.empty() || points.getNumNonZeros() != edgePattern.getNumNonZeros() ||
      static_cast<size_t>(points.getOuterSize()) != nPoint) {
    SU2_MPI::Error("The point adjacency of the geometry is not set up or is inconsistent.", CURRENT_FUNCTION);
  }

  const auto& normals = geometry.edges->GetNormal();
  if (normals.rows() != nEdge || normals.cols() != nDim) {
    SU2_MPI::Error("The edge normals of the geometry are not set up or are inconsistent.", CURRENT_FUNCTION);
  }

  const size_t nNonZero = points.getNumNonZeros();
  Send(nbrOuter, points.outerPtr(), nPoint + 1, "nbrOuter");
  Send(nbrIdx, points.innerIdx(), nNonZero, "nbrIdx");
  Send(edgeIdx, edgePattern.innerIdx(), nNonZero, "edgeIdx");
  Send(edgeNormal, normals.data(), nEdge * nDim, "edgeNormal");
  Send(volume, &nodes->GetVolume(0), nPoint, "volume");
  Send(periodicVolume, &nodes->GetPeriodicVolume(0), nPoint, "periodicVolume");

  hostDomain.resize(nPoint);
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) hostDomain[iPoint] = nodes->GetDomain(iPoint) ? 1 : 0;
  Send(domain, hostDomain.data(), nPoint, "domain");

  /*--- Markers, the vector is sized first as the staging buffers must not move once a mirror is attached
   *    (the mirrors page-lock them). ---*/
  markers.resize(geometry.GetnMarker());
  for (size_t iMarker = 0; iMarker < markers.size(); ++iMarker) {
    auto& marker = markers[iMarker];
    const size_t nVertex = geometry.GetnVertex(iMarker);
    marker.kindBC = config.GetMarker_All_KindBC(iMarker);
    marker.hostNode.resize(nVertex);
    marker.hostNormal.resize(nVertex * nDim);
    for (size_t iVertex = 0; iVertex < nVertex; ++iVertex) {
      const auto* vertex = geometry.vertex[iMarker][iVertex];
      marker.hostNode[iVertex] = vertex->GetNode();
      for (size_t iDim = 0; iDim < nDim; ++iDim) marker.hostNormal[iVertex * nDim + iDim] = vertex->GetNormal(iDim);
    }
    const auto tag = "marker" + std::to_string(iMarker);
    Send(marker.node, marker.hostNode.data(), nVertex, tag + ".node");
    Send(marker.normal, marker.hostNormal.data(), nVertex * nDim, tag + ".normal");
  }
}

CGeometryDevice::CGeometryDevice(const CGeometry& geometry, const CConfig& config)
    : nPoint(geometry.GetnPoint()),
      nPointDomain(geometry.GetnPointDomain()),
      nEdge(geometry.GetnEdge()),
      nDim(geometry.GetnDim()) {
  try {
    Upload(geometry, config);
  } catch (const std::exception& e) {
    SU2_MPI::Error(std::string("Could not copy the geometry to the GPU: ") + e.what(), CURRENT_FUNCTION);
  }

  if (SU2_MPI::GetRank() == MASTER_NODE) {
    std::cout << "GPU geometry snapshot: " << bytes << " bytes uploaded (" << std::fixed << std::setprecision(2)
              << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB)." << std::endl;
  }
}

CGeometryDevice::View CGeometryDevice::GetGeomView() const {
  View view;
  view.nbrOuter = nbrOuter.DeviceRead();
  view.nbrIdx = nbrIdx.DeviceRead();
  view.edgeIdx = edgeIdx.DeviceRead();
  view.normal = edgeNormal.DeviceRead();
  view.normalLd = nDim;
  view.volume = volume.DeviceRead();
  view.periodicVolume = periodicVolume.DeviceRead();
  return view;
}

CGeometryDevice::MarkerView CGeometryDevice::GetMarker(size_t iMarker) const {
  const auto& marker = markers.at(iMarker);
  MarkerView view;
  view.kindBC = marker.kindBC;
  view.nVertex = marker.hostNode.size();
  view.node = marker.node.DeviceRead();
  view.normal = marker.normal.DeviceRead();
  return view;
}

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
