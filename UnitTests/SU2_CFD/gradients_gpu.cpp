/*!
 * @file gradients_gpu.cpp
 * @brief Tests of the GPU version of the Green-Gauss gradient (volume and boundary marker contributions).
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

#include "../../Common/include/code_config.hpp"

#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "catch.hpp"
#include "../UnitQuadTestCase.hpp"
#include "../../Common/include/containers/container_decorators.hpp"
#include "../../SU2_CFD/include/gradients/computeGradientsGreenGauss.hpp"
#include "../../SU2_CFD/include/gradients/computeGradientsGreenGaussGPU.hpp"

namespace {

/*--- Same selection of markers as in the solver and the CPU implementation. ---*/
std::vector<char> SkippedMarkers(const CGeometry& geometry, const CConfig& config) {
  std::vector<char> skipped(geometry.GetnMarker());
  for (size_t iMarker = 0; iMarker < skipped.size(); ++iMarker) {
    const auto kind = config.GetMarker_All_KindBC(iMarker);
    skipped[iMarker] = (kind == INTERNAL_BOUNDARY) || (kind == NEARFIELD_BOUNDARY) || (kind == PERIODIC_BOUNDARY);
  }
  return skipped;
}

/*--- Compare the domain rows of two gradients byte for byte, the halo rows of the GPU one must be untouched. ---*/
template <size_t nDim>
void CompareBitwise(const CGeometry& geometry, size_t nVar, C3DDoubleMatrix& cpu, C3DDoubleMatrix& gpu,
                    double sentinel) {
  /*--- nVar is the number of variables stored in the containers, which may be more than the computed ones, the
   * entries that are not computed must still be the same (the sentinel). ---*/
  const size_t nPoint = geometry.GetnPoint(), nPointDomain = geometry.GetnPointDomain();
  const size_t nDomainEntries = nPointDomain * nVar * nDim;

  const bool same = std::memcmp(cpu.data(), gpu.data(), nDomainEntries * sizeof(su2double)) == 0;
  double maxRel = 0.0;
  size_t nDiff = 0;
  if (!same) {
    for (size_t k = 0; k < nDomainEntries; ++k) {
      const double a = cpu.data()[k], b = gpu.data()[k];
      if (std::memcmp(&a, &b, sizeof(double)) != 0) {
        ++nDiff;
        maxRel = std::max(maxRel, std::abs(a - b) / std::max(std::abs(a), 1e-300));
      }
    }
  }
  INFO("entries that differ: " << nDiff << " of " << nDomainEntries << ", max relative difference: " << maxRel);
  CHECK(same);

  /*--- Single rank grids have no halos, but the check is kept for the general case. ---*/
  for (size_t k = nDomainEntries; k < nPoint * nVar * nDim; ++k) CHECK(gpu.data()[k] == sentinel);
}

/*--- Run the CPU and the GPU version of the first part on a random field and compare the results. ---*/
template <size_t nDim>
void RunComparison(CGeometry& geometry, const CConfig& config, size_t nVar, size_t extraColumns, unsigned seed,
                   size_t extraGradVars = 0) {
  REQUIRE(geometry.GetnDim() == nDim);
  const size_t nPoint = geometry.GetnPoint();

  /*--- The field has more columns than variables of the gradient, as the primitives do. ---*/
  su2activematrix field(nPoint, nVar + extraColumns);
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-3.0, 7.0);
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint)
    for (size_t iVar = 0; iVar < field.cols(); ++iVar) field(iPoint, iVar) = dist(gen) * (1.0 + 1e3 * (iVar % 3 == 0));

  const double sentinel = 12345.678;
  /*--- The gradient containers can store more variables than are computed (like in the flow solvers). ---*/
  const size_t nStore = nVar + extraGradVars;
  C3DDoubleMatrix cpu(nPoint, nStore, nDim, sentinel);
  C3DDoubleMatrix gpu(nPoint, nStore, nDim, sentinel);

  detail::computeGradientsGreenGaussDomain<nDim>(geometry, config, field, 0, nVar, cpu);

  gg_gpu::GreenGaussDeviceData data;
  bool firstCall = false;
  gg_gpu::ComputeGradientsGreenGauss(data, geometry.GetDevice(&config), SkippedMarkers(geometry, config),
                                     field.data(), field.cols(), gpu.data(), nStore, nVar, 0, firstCall);
  CHECK(firstCall);

  CompareBitwise<nDim>(geometry, nStore, cpu, gpu, sentinel);

  /*--- A second call (the values changed on the host) reuses the mirrors. ---*/
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) field(iPoint, 0) = dist(gen);
  detail::computeGradientsGreenGaussDomain<nDim>(geometry, config, field, 0, nVar, cpu);
  gg_gpu::ComputeGradientsGreenGauss(data, geometry.GetDevice(&config), SkippedMarkers(geometry, config),
                                     field.data(), field.cols(), gpu.data(), nStore, nVar, 0, firstCall);
  CHECK(!firstCall);
  CompareBitwise<nDim>(geometry, nStore, cpu, gpu, sentinel);
}

}  // namespace

TEST_CASE("Green-Gauss gradient on the GPU is bit-identical to the CPU (3D)", "[CUDA]") {
  UnitQuadTestCase testCase;
  testCase.InitConfig();
  testCase.InitGeometry();
  REQUIRE(testCase.geometry->GetnDim() == 3);

  SECTION("1 variable") { RunComparison<3>(*testCase.geometry, *testCase.config, 1, 0, 1); }
  SECTION("5 variables, padded field") { RunComparison<3>(*testCase.geometry, *testCase.config, 5, 2, 2); }
  SECTION("7 variables") { RunComparison<3>(*testCase.geometry, *testCase.config, 7, 0, 3); }
  SECTION("3 variables in a gradient that stores 5") { RunComparison<3>(*testCase.geometry, *testCase.config, 3, 2, 7, 2); }
}

TEST_CASE("Green-Gauss gradient on the GPU is bit-identical to the CPU (2D)", "[CUDA]") {
  UnitQuadTestCase testCase;
  testCase.config_options =
      "SOLVER= NAVIER_STOKES\n"
      "MESH_FORMAT= RECTANGLE\n"
      "INIT_OPTION= TD_CONDITIONS\n"
      "MARKER_HEATFLUX= (y_minus, 0.0, y_plus, 0.0)\n"
      "MARKER_FAR= (x_minus, x_plus)\n"
      "MESH_BOX_SIZE= 21,13,1\n"
      "MESH_BOX_LENGTH= 2,1,1\n"
      "MESH_BOX_OFFSET= 0,0,0\n";
  testCase.InitConfig();
  testCase.InitGeometry();
  REQUIRE(testCase.geometry->GetnDim() == 2);

  SECTION("1 variable") { RunComparison<2>(*testCase.geometry, *testCase.config, 1, 0, 4); }
  SECTION("4 variables, padded field") { RunComparison<2>(*testCase.geometry, *testCase.config, 4, 3, 5); }
  SECTION("6 variables") { RunComparison<2>(*testCase.geometry, *testCase.config, 6, 0, 6); }
  SECTION("3 variables in a gradient that stores 4") { RunComparison<2>(*testCase.geometry, *testCase.config, 3, 5, 8, 1); }
}

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
