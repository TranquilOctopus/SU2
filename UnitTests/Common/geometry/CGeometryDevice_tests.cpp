/*!
 * @file CGeometryDevice_tests.cpp
 * @brief Unit tests for the device snapshot of the geometry (needs a device).
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

#include "catch.hpp"

#include "../../../Common/include/code_config.hpp"

#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <cstring>
#include <type_traits>
#include <vector>

#include "../../UnitQuadTestCase.hpp"
#include "../../../Common/include/geometry/CGeometryDevice.hpp"
#include "../../../SU2_CFD/include/gradients/computeGradientsGreenGauss.hpp"

namespace {

/*--- Copy a device array to a vector. ---*/
template <class T>
std::vector<T> Download(const T* devPtr, size_t n) {
  std::vector<T> out(n);
  if (n > 0) {
    REQUIRE(devPtr != nullptr);
    REQUIRE(cudaMemcpy(out.data(), devPtr, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  return out;
}

/*--- Compare a device array with host data, byte for byte. ---*/
template <class T>
bool SameBytes(const T* devPtr, const T* hostPtr, size_t n) {
  const auto dev = Download(devPtr, n);
  return n == 0 || std::memcmp(dev.data(), hostPtr, n * sizeof(T)) == 0;
}

}  // namespace

TEST_CASE("Device snapshot of the geometry", "[CUDA]") {
  UnitQuadTestCase testCase;
  testCase.InitConfig();
  testCase.InitGeometry();

  const auto& geometry = *testCase.geometry;
  const auto& config = *testCase.config;
  const auto* nodes = geometry.nodes;

  const auto& device = geometry.GetDevice(&config);

  SECTION("Sizes") {
    CHECK(device.GetnPoint() == geometry.GetnPoint());
    CHECK(device.GetnPointDomain() == geometry.GetnPointDomain());
    CHECK(device.GetnEdge() == geometry.GetnEdge());
    CHECK(device.GetnDim() == geometry.GetnDim());
    CHECK(device.GetnMarker() == geometry.GetnMarker());
    CHECK(device.GetBytes() > 0);
  }

  SECTION("The snapshot is built once") { CHECK(&geometry.GetDevice(&config) == &device); }

  SECTION("Adjacency, edges, normals and volumes") {
    const auto host = detail::makeGreenGaussGeomView(geometry);
    const auto view = device.GetGeomView();
    const size_t nPoint = geometry.GetnPoint();
    const size_t nEdge = geometry.GetnEdge();
    const size_t nDim = geometry.GetnDim();

    /*--- The device view is the same type as the host one, with the same strides. ---*/
    static_assert(std::is_same<std::remove_const<decltype(host)>::type, std::remove_const<decltype(view)>::type>::value,
                  "The host and device views must be the same type.");
    CHECK(view.normalLd == host.normalLd);
    CHECK(view.normalLd == nDim);

    const size_t nNonZero = host.nbrOuter[nPoint];
    REQUIRE(nNonZero > 0);

    CHECK(SameBytes(view.nbrOuter, host.nbrOuter, nPoint + 1));
    CHECK(SameBytes(view.nbrIdx, host.nbrIdx, nNonZero));
    CHECK(SameBytes(view.edgeIdx, host.edgeIdx, nNonZero));
    CHECK(SameBytes(view.normal, host.normal, nEdge * nDim));
    CHECK(SameBytes(view.volume, host.volume, nPoint));
    CHECK(SameBytes(view.periodicVolume, host.periodicVolume, nPoint));

    /*--- Check against the accessors too, not only against the raw pointers of the host view. ---*/
    const auto nbrOuter = Download(view.nbrOuter, nPoint + 1);
    const auto nbrIdx = Download(view.nbrIdx, nNonZero);
    const auto edgeIdx = Download(view.edgeIdx, nNonZero);
    const auto normal = Download(view.normal, nEdge * nDim);
    const auto volume = Download(view.volume, nPoint);
    for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) {
      const auto& points = nodes->GetPoints();
      REQUIRE(nbrOuter[iPoint + 1] - nbrOuter[iPoint] == points.getNumNonZeros(iPoint));
      size_t k = nbrOuter[iPoint];
      auto edge = nodes->GetEdges(iPoint).begin();
      for (auto jPoint : nodes->GetPoints(iPoint)) {
        CHECK(nbrIdx[k] == jPoint);
        CHECK(edgeIdx[k] == *edge);
        const auto* hostNormal = geometry.edges->GetNormal(edgeIdx[k]);
        CHECK(std::memcmp(&normal[edgeIdx[k] * nDim], hostNormal, nDim * sizeof(su2double)) == 0);
        ++k, ++edge;
      }
      CHECK(std::memcmp(&volume[iPoint], &nodes->GetVolume(iPoint), sizeof(su2double)) == 0);
    }
  }

  SECTION("Domain flags") {
    const size_t nPoint = geometry.GetnPoint();
    const auto flags = Download(device.GetDomain(), nPoint);
    size_t nDomain = 0;
    for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) {
      CHECK((flags[iPoint] != 0) == nodes->GetDomain(iPoint));
      nDomain += flags[iPoint];
    }
    CHECK(nDomain > 0);
  }

  SECTION("Markers") {
    const size_t nDim = geometry.GetnDim();
    size_t nVertexTotal = 0;
    for (size_t iMarker = 0; iMarker < geometry.GetnMarker(); ++iMarker) {
      const auto marker = device.GetMarker(iMarker);
      const size_t nVertex = geometry.GetnVertex(iMarker);
      CHECK(marker.kindBC == config.GetMarker_All_KindBC(iMarker));
      REQUIRE(marker.nVertex == nVertex);
      nVertexTotal += nVertex;

      const auto node = Download(marker.node, nVertex);
      const auto normal = Download(marker.normal, nVertex * nDim);
      for (size_t iVertex = 0; iVertex < nVertex; ++iVertex) {
        const auto* vertex = geometry.vertex[iMarker][iVertex];
        CHECK(node[iVertex] == vertex->GetNode());
        for (size_t iDim = 0; iDim < nDim; ++iDim) {
          const su2double hostValue = vertex->GetNormal(iDim);
          CHECK(std::memcmp(&normal[iVertex * nDim + iDim], &hostValue, sizeof(su2double)) == 0);
        }
      }
    }
    CHECK(nVertexTotal > 0);
  }
}

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
