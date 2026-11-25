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

  CFlowDeviceData data;
  bool firstCall = false;
  gg_gpu::ComputeGradientsGreenGauss(data, geometry.GetDevice(&config), SkippedMarkers(geometry, config),
                                     field.data(), field.cols(), gpu.data(), nStore, nVar, -1, 0, firstCall);
  CHECK(firstCall);
  const size_t bytesPrim = nPoint * field.cols() * sizeof(double);
  const size_t bytesGrad = geometry.GetnPointDomain() * nStore * nDim * sizeof(double);
  const size_t bytesComputed = geometry.GetnPointDomain() * nVar * nDim * sizeof(double);
  CHECK(data.BytesUploaded() == bytesPrim);
  /*--- Only the computed variables come back (the others keep the sentinel of the host, checked below). ---*/
  CHECK(data.BytesDownloaded() == (extraGradVars ? bytesComputed : bytesGrad));

  CompareBitwise<nDim>(geometry, nStore, cpu, gpu, sentinel);

  /*--- A second call without a change reuses the device primitives: nothing is uploaded. ---*/
  gg_gpu::ComputeGradientsGreenGauss(data, geometry.GetDevice(&config), SkippedMarkers(geometry, config),
                                     field.data(), field.cols(), gpu.data(), nStore, nVar, -1, 0, firstCall);
  CHECK(!firstCall);
  CHECK(data.BytesUploaded() == bytesPrim);
  CompareBitwise<nDim>(geometry, nStore, cpu, gpu, sentinel);

  /*--- A third call (the values changed on the host, and the owner says so) uploads again. ---*/
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) field(iPoint, 0) = dist(gen);
  data.PrimitivesModified();
  detail::computeGradientsGreenGaussDomain<nDim>(geometry, config, field, 0, nVar, cpu);
  gg_gpu::ComputeGradientsGreenGauss(data, geometry.GetDevice(&config), SkippedMarkers(geometry, config),
                                     field.data(), field.cols(), gpu.data(), nStore, nVar, -1, 0, firstCall);
  CHECK(!firstCall);
  CompareBitwise<nDim>(geometry, nStore, cpu, gpu, sentinel);
  CHECK(data.BytesUploaded() == 2 * bytesPrim);
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


namespace {

/*--- Number of vertices of the symmetry and Euler wall markers, and of the points that have a modified normal
 *    (points shared by several symmetry markers), so that the test knows that it exercises both paths. ---*/
void CountSymmetry(const CGeometry& geometry, const CConfig& config, size_t& nVertices, size_t& nModified) {
  nVertices = nModified = 0;
  for (size_t iMarker = 0; iMarker < geometry.GetnMarker(); ++iMarker) {
    const auto kind = config.GetMarker_All_KindBC(iMarker);
    if (kind != SYMMETRY_PLANE && kind != EULER_WALL) continue;
    nVertices += geometry.GetnVertex(iMarker);
    nModified += geometry.symmetryNormals[iMarker].size();
  }
}

/*--- The correction alone on random gradients (all variables random, the gradient container can store more
 *    variables than are corrected), and then the complete Green-Gauss gradient (the first and second parts). ---*/
template <size_t nDim>
void RunSymmetryComparison(CGeometry& geometry, const CConfig& config, size_t nVar, int idxVel, size_t extraGradVars,
                           unsigned seed) {
  REQUIRE(geometry.GetnDim() == nDim);
  const size_t nPoint = geometry.GetnPoint(), nPointDomain = geometry.GetnPointDomain();
  const size_t nStore = nVar + extraGradVars;

  size_t nVertices, nModified;
  CountSymmetry(geometry, config, nVertices, nModified);
  REQUIRE(nVertices > 0);
  REQUIRE(nModified > 0);

  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-3.0, 7.0);

  /*--- Part 1: correction of random gradients. ---*/
  {
    C3DDoubleMatrix orig(nPoint, nStore, nDim, 0.0);
    for (size_t k = 0; k < nPoint * nStore * nDim; ++k) orig.data()[k] = dist(gen);
    C3DDoubleMatrix cpu(nPoint, nStore, nDim, 0.0);
    C3DDoubleMatrix gpu(nPoint, nStore, nDim, 0.0);
    std::memcpy(cpu.data(), orig.data(), nPoint * nStore * nDim * sizeof(double));
    std::memcpy(gpu.data(), orig.data(), nPoint * nStore * nDim * sizeof(double));

    correctGradientsSymmetry<nDim>(geometry, config, 0, nVar, idxVel, cpu);

    CFlowDeviceData data;
    data.Attach(data.gradient[0], gpu.data(), nPointDomain * nStore * nDim, "test gradient");
    data.gradient[0].ToDevice();
    gg_gpu::CorrectGradientsSymmetry(geometry.GetDevice(&config), nDim, data.gradient[0].DeviceWrite(), nStore, 0,
                                     nVar, idxVel);
    data.gradient[0].DeviceModified();
    data.gradient[0].ToHost();

    /*--- The correction did something, and left the variables outside of the range alone. ---*/
    CHECK(std::memcmp(cpu.data(), orig.data(), nPoint * nStore * nDim * sizeof(double)) != 0);
    CompareBitwise<nDim>(geometry, nStore, cpu, gpu, 0.0);
  }

  /*--- Part 2: the complete gradient, CPU (both parts) against GPU (first and second part). ---*/
  {
    su2activematrix field(nPoint, nVar + 2);
    for (size_t iPoint = 0; iPoint < nPoint; ++iPoint)
      for (size_t iVar = 0; iVar < field.cols(); ++iVar) field(iPoint, iVar) = dist(gen);

    const double sentinel = 12345.678;
    C3DDoubleMatrix cpu(nPoint, nStore, nDim, sentinel);
    C3DDoubleMatrix gpu(nPoint, nStore, nDim, sentinel);

    detail::computeGradientsGreenGauss<nDim>(nullptr, MPI_QUANTITIES::SOLUTION, PERIODIC_NONE, geometry, config,
                                             field, 0, nVar, idxVel, cpu);

    CFlowDeviceData data;
    bool firstCall = false;
    gg_gpu::ComputeGradientsGreenGauss(data, geometry.GetDevice(&config), SkippedMarkers(geometry, config),
                                       field.data(), field.cols(), gpu.data(), nStore, nVar, idxVel, 0, firstCall);
    CHECK(firstCall);
    CompareBitwise<nDim>(geometry, nStore, cpu, gpu, sentinel);

    /*--- A second call (the gradient is overwritten, the primitives are reused) gives the same result. ---*/
    gg_gpu::ComputeGradientsGreenGauss(data, geometry.GetDevice(&config), SkippedMarkers(geometry, config),
                                       field.data(), field.cols(), gpu.data(), nStore, nVar, idxVel, 0, firstCall);
    CHECK(!firstCall);
    CompareBitwise<nDim>(geometry, nStore, cpu, gpu, sentinel);
  }
}

}  // namespace

TEST_CASE("Gradient symmetry correction on the GPU is bit-identical to the CPU (3D)", "[CUDA]") {
  UnitQuadTestCase testCase;
  testCase.config_options =
      "SOLVER= EULER\n"
      "MESH_FORMAT= BOX\n"
      "INIT_OPTION= TD_CONDITIONS\n"
      "MARKER_SYM= (x_minus, z_minus)\n"
      "MARKER_EULER= (y_minus, x_plus)\n"
      "MARKER_FAR= (y_plus, z_plus)\n"
      "MESH_BOX_SIZE= 9,7,6\n"
      "MESH_BOX_LENGTH= 2,1,1\n"
      "MESH_BOX_OFFSET= 0,0,0\n";
  testCase.InitConfig();
  testCase.InitGeometry();
  REQUIRE(testCase.geometry->GetnDim() == 3);

  SECTION("scalars only") { RunSymmetryComparison<3>(*testCase.geometry, *testCase.config, 3, -1, 0, 1); }
  SECTION("5 variables, velocity at 1") { RunSymmetryComparison<3>(*testCase.geometry, *testCase.config, 5, 1, 0, 2); }
  SECTION("4 variables, velocity at 0") { RunSymmetryComparison<3>(*testCase.geometry, *testCase.config, 4, 0, 0, 3); }
  SECTION("7 variables, velocity at 2") { RunSymmetryComparison<3>(*testCase.geometry, *testCase.config, 7, 2, 0, 4); }
  SECTION("5 variables in a gradient that stores 8") {
    RunSymmetryComparison<3>(*testCase.geometry, *testCase.config, 5, 1, 3, 5);
  }
}

TEST_CASE("Gradient symmetry correction on the GPU is bit-identical to the CPU (2D)", "[CUDA]") {
  UnitQuadTestCase testCase;
  testCase.config_options =
      "SOLVER= EULER\n"
      "MESH_FORMAT= RECTANGLE\n"
      "INIT_OPTION= TD_CONDITIONS\n"
      "MARKER_SYM= (x_minus)\n"
      "MARKER_EULER= (y_minus)\n"
      "MARKER_FAR= (x_plus, y_plus)\n"
      "MESH_BOX_SIZE= 21,13,1\n"
      "MESH_BOX_LENGTH= 2,1,1\n"
      "MESH_BOX_OFFSET= 0,0,0\n";
  testCase.InitConfig();
  testCase.InitGeometry();
  REQUIRE(testCase.geometry->GetnDim() == 2);

  SECTION("scalars only") { RunSymmetryComparison<2>(*testCase.geometry, *testCase.config, 3, -1, 0, 6); }
  SECTION("4 variables, velocity at 1") { RunSymmetryComparison<2>(*testCase.geometry, *testCase.config, 4, 1, 0, 7); }
  SECTION("6 variables, velocity at 3") { RunSymmetryComparison<2>(*testCase.geometry, *testCase.config, 6, 3, 0, 8); }
  SECTION("4 variables in a gradient that stores 6") {
    RunSymmetryComparison<2>(*testCase.geometry, *testCase.config, 4, 1, 2, 9);
  }
}

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
