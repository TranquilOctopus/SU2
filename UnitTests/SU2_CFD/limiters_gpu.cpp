/*!
 * @file limiters_gpu.cpp
 * @brief Tests of the GPU version of the slope limiters of the primitive variables (Barth-Jespersen, Venkatakrishnan,
 *        Nishikawa R3, R4 and R5, Venkatakrishnan-Wang) against the CPU implementation.
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "catch.hpp"
#include "../UnitQuadTestCase.hpp"
#include "../../Common/include/containers/container_decorators.hpp"
#include "../../SU2_CFD/include/gradients/computeGradientsLeastSquares.hpp"
#include "../../SU2_CFD/include/gradients/computeGradientsLeastSquaresGPU.hpp"
#include "../../SU2_CFD/include/limiters/computeLimiters.hpp"
#include "../../SU2_CFD/include/limiters/computeLimitersGPU.hpp"

/*============================================================================================================
 * Slope limiters: CPU against GPU.
 *
 * Like the least-squares gradient, and unlike the Green-Gauss gradient, the limiter kernels are not guaranteed to be
 * bit-identical to the CPU, the policy is a relative difference of at most kLimiterTolerance per entry (the number of
 * entries that differ in bits and the worst differences are printed, see the header of computeLimitersGPU.cu for
 * the numbers that were observed). The host (-O3 -march=native -ffast-math) is free to rewrite the limiter
 * functions (reciprocals of repeated denominators, the expansion of pow(x, n) with an integer n, the choice of the
 * products that are contracted into FMAs), the device compiler (-fmad=true, IEEE division) computes the operations
 * as written. The global minimum and maximum used by Venkatakrishnan-Wang are exact operations and must be
 * bit-identical.
 *============================================================================================================*/

namespace {

constexpr double kLimiterTolerance = 1e-13;

/*--- Smooth fields with noise, so that there are extrema (limiter below 1) and smooth regions, different amplitudes
 *    and offsets per variable (the global ranges of Venkatakrishnan-Wang differ). ---*/
template <size_t nDim>
void FillField(const CGeometry& geometry, su2activematrix& field, unsigned seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> noise(-1.0, 1.0);
  for (size_t iPoint = 0; iPoint < field.rows(); ++iPoint) {
    const auto* x = geometry.nodes->GetCoord(iPoint);
    for (size_t iVar = 0; iVar < field.cols(); ++iVar) {
      const double amp = 1.0 + 3.0 * (iVar % 3), off = 10.0 * (iVar % 2);
      double s = 0.0;
      for (size_t iDim = 0; iDim < nDim; ++iDim) s += std::sin(1.7 * x[iDim] + 0.9 * iVar + iDim);
      field(iPoint, iVar) = off + amp * (s + 0.15 * noise(gen));
    }
  }
}

struct Difference {
  size_t nDiff = 0, nEntries = 0;
  double maxRel = 0.0, maxAbs = 0.0;
};

Difference Compare(const CGeometry& geometry, size_t nVar, const su2activematrix& cpu, const su2activematrix& gpu) {
  Difference d;
  for (size_t iPoint = 0; iPoint < geometry.GetnPointDomain(); ++iPoint) {
    for (size_t iVar = 0; iVar < nVar; ++iVar) {
      const double a = cpu(iPoint, iVar), b = gpu(iPoint, iVar);
      ++d.nEntries;
      if (std::memcmp(&a, &b, sizeof(double)) != 0) {
        ++d.nDiff;
        d.maxAbs = std::max(d.maxAbs, std::abs(a - b));
        d.maxRel = std::max(d.maxRel, std::abs(a - b) / std::max(std::abs(a), std::abs(b)));
      }
    }
  }
  return d;
}

/*--- The CPU limiter, with the gradient that the GPU is given (no solver: no communications). ---*/
template <class Gradient>
void CpuLimiter(CGeometry& geometry, const CConfig& config, LIMITER kind, size_t nVar, su2double kappa,
                const su2activematrix& field, const Gradient& gradient, su2activematrix& fieldMin,
                su2activematrix& fieldMax, su2activematrix& limiter) {
  computeLimiters(kind, nullptr, MPI_QUANTITIES::PRIMITIVE_LIMITER, PERIODIC_NONE, PERIODIC_NONE, geometry, config, 0,
                  nVar, kappa, field, gradient, fieldMin, fieldMax, limiter);
}

/*--- Run the CPU and the GPU limiter on the same field and gradient and compare the results.
 *    extraColumns: the field (and the limiter) have more columns than variables, like the primitives and the
 *    limiter of the solvers. extraGradVars: the gradient stores more variables than are computed. ---*/
template <size_t nDim>
void RunLimiterComparison(CGeometry& geometry, const CConfig& config, LIMITER kind, size_t nVar, size_t extraColumns,
                          size_t extraGradVars, su2double kappa, unsigned seed) {
  REQUIRE(geometry.GetnDim() == nDim);
  const size_t nPoint = geometry.GetnPoint(), nPointDomain = geometry.GetnPointDomain();

  su2activematrix field(nPoint, nVar + extraColumns);
  FillField<nDim>(geometry, field, seed);

  /*--- A realistic reconstruction gradient (least squares on the CPU), stored in a container that is wider. ---*/
  const size_t nStore = nVar + extraGradVars;
  const double sentinel = 12345.678;
  C3DDoubleMatrix gradient(nPoint, nStore, nDim, sentinel);
  C3DDoubleMatrix rMatrix(nPoint, nDim, nDim, 0.0);
  computeGradientsLeastSquares(nullptr, MPI_QUANTITIES::SOLUTION, PERIODIC_NONE, geometry, config, true, field, 0,
                               nVar, -1, gradient, rMatrix);

  /*--- The limiter container has the columns of the field, the limiter is computed for the first nVar. ---*/
  su2activematrix fieldMin(nPoint, field.cols()), fieldMax(nPoint, field.cols());
  su2activematrix cpu(nPoint, field.cols()), gpu(nPoint, field.cols());
  cpu = su2double(sentinel);
  gpu = su2double(sentinel);

  CpuLimiter(geometry, config, kind, nVar, kappa, field, gradient, fieldMin, fieldMax, cpu);

  CFlowDeviceData data;
  bool firstCall = false;
  lim_gpu::ComputeLimiters(data, geometry, geometry.GetDevice(&config), config, kind, kappa, field.data(), field.cols(),
                           gradient.data(), nStore, nVar, gpu.data(), gpu.cols(), firstCall);
  CHECK(firstCall);

  const bool isVW = (kind == LIMITER::VENKATAKRISHNAN_WANG);
  const size_t bytesPrim = nPoint * field.cols() * sizeof(double);
  const size_t bytesGrad = nPoint * nStore * nDim * sizeof(double);
  /*--- Both inputs come from the host in this test (the primitives lazily, and so the gradient, which did not
   * come from the GPU). The limiter comes back once, plus the 2 values per variable of Venkatakrishnan-Wang. ---*/
  CHECK(data.BytesUploaded() == bytesPrim + bytesGrad);
  CHECK(data.BytesDownloaded() == nPoint * nVar * sizeof(double) + (isVW ? 2 * nVar * sizeof(double) : 0));
  /*--- Only the computed variables are copied back when the limiter container has more columns, the other entries
   * keep their values. Without extra columns the whole matrix is copied (all the entries are computed). ---*/
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint)
    for (size_t iVar = nVar; iVar < field.cols(); ++iVar) CHECK(gpu(iPoint, iVar) == sentinel);

  const auto d = Compare(geometry, nVar, cpu, gpu);

  /*--- How many limiters are neither 0 nor 1: the test is only meaningful if the functions are exercised. ---*/
  size_t nPartial = 0;
  for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint)
    for (size_t iVar = 0; iVar < nVar; ++iVar) nPartial += (cpu(iPoint, iVar) > 1e-12 && cpu(iPoint, iVar) < 1.0 - 1e-12);
  CHECK(nPartial > nPointDomain * nVar / 20);

  std::cout << std::scientific << std::setprecision(3) << "Limiter GPU vs CPU (" << nDim << "D, kind "
            << lim_gpu::Name(kind) << ", " << nVar << " vars, kappa " << kappa << "): " << d.nDiff << " of "
            << d.nEntries << " entries differ in bits, max relative difference " << d.maxRel << ", max absolute "
            << d.maxAbs << ", " << nPartial << " limiters strictly between 0 and 1" << std::endl;
  INFO("entries that differ: " << d.nDiff << " of " << d.nEntries << ", max relative difference " << d.maxRel);
  CHECK(d.maxRel <= kLimiterTolerance);

  /*--- Second call: nothing is uploaded again, same result. ---*/
  const size_t up = data.BytesUploaded();
  lim_gpu::ComputeLimiters(data, geometry, geometry.GetDevice(&config), config, kind, kappa, field.data(), field.cols(),
                           gradient.data(), nStore, nVar, gpu.data(), gpu.cols(), firstCall);
  CHECK(!firstCall);
  CHECK(data.BytesUploaded() == up);
  CHECK(Compare(geometry, nVar, cpu, gpu).maxRel <= kLimiterTolerance);

  /*--- The field changed on the host and the owner says so: uploaded again (the primitives only). ---*/
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) field(iPoint, 0) = field(iPoint, 0) * 1.5 + 0.1 * iPoint;
  data.PrimitivesModified();
  CpuLimiter(geometry, config, kind, nVar, kappa, field, gradient, fieldMin, fieldMax, cpu);
  lim_gpu::ComputeLimiters(data, geometry, geometry.GetDevice(&config), config, kind, kappa, field.data(), field.cols(),
                           gradient.data(), nStore, nVar, gpu.data(), gpu.cols(), firstCall);
  CHECK(data.BytesUploaded() == up + bytesPrim);
  CHECK(Compare(geometry, nVar, cpu, gpu).maxRel <= kLimiterTolerance);
}

/*--- The pipeline of the solver: the gradient is computed on the GPU first and the limiter reads it from the
 *    device, nothing but the primitives is uploaded. The CPU reference uses the gradient that came back.
 *    which = 1: the gradient is the reconstruction gradient. which = 0: the flow variables alias the reconstruction
 *    gradient to the viscous one when the methods are the same, so the limiter reads the mirror of the viscous
 *    gradient (the one that the GPU kept current), and the gradient is recomputed (different values) before a
 *    second limiter, which must see it. ---*/
template <size_t nDim>
void RunChainedComparison(CGeometry& geometry, const CConfig& config, LIMITER kind, size_t nVar, unsigned seed,
                          int which) {
  const size_t nPoint = geometry.GetnPoint();

  su2activematrix field(nPoint, nVar + 2);
  FillField<nDim>(geometry, field, seed);

  C3DDoubleMatrix gradient(nPoint, nVar, nDim, 0.0);
  su2activematrix fieldMin(nPoint, field.cols()), fieldMax(nPoint, field.cols());
  su2activematrix gpu(nPoint, nVar);

  CFlowDeviceData data;
  bool firstCall = false;
  ls_gpu::ComputeGradientsLeastSquares(data, geometry.GetDevice(&config), true, field.data(), field.cols(),
                                       gradient.data(), nVar, nVar, -1, which, firstCall);
  const size_t up = data.BytesUploaded();
  CHECK(up == nPoint * field.cols() * sizeof(double));

  lim_gpu::ComputeLimiters(data, geometry, geometry.GetDevice(&config), config, kind, 0.0, field.data(), field.cols(),
                           gradient.data(), nVar, nVar, gpu.data(), gpu.cols(), firstCall);
  CHECK(firstCall);
  /*--- Neither the primitives nor the gradient are uploaded again. ---*/
  CHECK(data.BytesUploaded() == up);

  /*--- The limiter container of the CPU has the columns of the field, use the first nVar. ---*/
  su2activematrix cpuFull(nPoint, field.cols());
  CpuLimiter(geometry, config, kind, nVar, 0.0, field, gradient, fieldMin, fieldMax, cpuFull);
  const auto d = Compare(geometry, nVar, cpuFull, gpu);
  std::cout << std::scientific << std::setprecision(3) << "Limiter GPU after the GPU gradient (" << nDim << "D, kind "
            << lim_gpu::Name(kind) << ", gradient " << which << "): " << d.nDiff << " of " << d.nEntries
            << " entries differ in bits, max relative difference " << d.maxRel << std::endl;
  CHECK(d.maxRel <= kLimiterTolerance);

  /*--- New field and gradient on the GPU, then the limiter again: no copy of the gradient to the device. ---*/
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) field(iPoint, 1 % nVar) = -2.0 * field(iPoint, 1 % nVar) + 0.3;
  data.PrimitivesModified();
  ls_gpu::ComputeGradientsLeastSquares(data, geometry.GetDevice(&config), true, field.data(), field.cols(),
                                       gradient.data(), nVar, nVar, -1, which, firstCall);
  const size_t up2 = data.BytesUploaded();
  lim_gpu::ComputeLimiters(data, geometry, geometry.GetDevice(&config), config, kind, 0.0, field.data(), field.cols(),
                           gradient.data(), nVar, nVar, gpu.data(), gpu.cols(), firstCall);
  CHECK(data.BytesUploaded() == up2);
  CpuLimiter(geometry, config, kind, nVar, 0.0, field, gradient, fieldMin, fieldMax, cpuFull);
  CHECK(Compare(geometry, nVar, cpuFull, gpu).maxRel <= kLimiterTolerance);
}

/*--- The extrema that the Venkatakrishnan-Wang limiter needs, against a plain loop on the host. They are exact. ---*/
void RunMinMaxCheck(CGeometry& geometry, const CConfig& config, size_t nVar, size_t varBegin, unsigned seed) {
  const size_t nPoint = geometry.GetnPoint(), nPointDomain = geometry.GetnPointDomain();
  std::mt19937_64 gen(seed);
  std::normal_distribution<double> dist(0.0, 1e3);
  su2activematrix field(nPoint, varBegin + nVar + 1);
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint)
    for (size_t iVar = 0; iVar < field.cols(); ++iVar) field(iPoint, iVar) = dist(gen);
  /*--- The extrema of the halo points must not count (there are none on one rank), put huge values in the last
   * columns that are not requested instead. ---*/
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) field(iPoint, field.cols() - 1) = 1e300;

  CFlowDeviceData data;
  data.Attach(data.primitives, field.data(), field.rows() * field.cols(), "field");
  data.primitives.ToDevice();

  const double largeNum = 0.1 * std::numeric_limits<double>::max();
  std::vector<double> lo(nVar), hi(nVar);
  lim_gpu::ReduceFieldMinMax(geometry.GetDevice(&config), data.primitives.DeviceRead(), field.cols(), varBegin,
                             varBegin + nVar, largeNum, lo.data(), hi.data());
  for (size_t iVar = 0; iVar < nVar; ++iVar) {
    double a = largeNum, b = -largeNum;
    for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint) {
      a = std::min(a, field(iPoint, varBegin + iVar));
      b = std::max(b, field(iPoint, varBegin + iVar));
    }
    CHECK(std::memcmp(&a, &lo[iVar], sizeof(double)) == 0);
    CHECK(std::memcmp(&b, &hi[iVar], sizeof(double)) == 0);
  }
}

/*--- A large Venkatakrishnan coefficient (the reference length is 1) puts the Venkatakrishnan(-Wang) and Nishikawa limiters in
 *    their smooth range for the field amplitudes of the tests. ---*/
const char* kLimiterConfig3D =
    "SOLVER= EULER\n"
    "MESH_FORMAT= BOX\n"
    "INIT_OPTION= TD_CONDITIONS\n"
    "MARKER_SYM= (x_minus, z_minus)\n"
    "MARKER_EULER= (y_minus, x_plus)\n"
    "MARKER_FAR= (y_plus, z_plus)\n"
    "MESH_BOX_SIZE= 11,9,8\n"
    "MESH_BOX_LENGTH= 2,1,1\n"
    "MESH_BOX_OFFSET= 0,0,0\n"
    "VENKAT_LIMITER_COEFF= 0.2\n";

const char* kLimiterConfig2D =
    "SOLVER= EULER\n"
    "MESH_FORMAT= RECTANGLE\n"
    "INIT_OPTION= TD_CONDITIONS\n"
    "MARKER_SYM= (x_minus)\n"
    "MARKER_EULER= (y_minus)\n"
    "MARKER_FAR= (x_plus, y_plus)\n"
    "MESH_BOX_SIZE= 31,19,1\n"
    "MESH_BOX_LENGTH= 2,1,1\n"
    "MESH_BOX_OFFSET= 0,0,0\n"
    "VENKAT_LIMITER_COEFF= 0.2\n";

void RunAllSections3D(LIMITER kind, unsigned seed) {
  UnitQuadTestCase testCase;
  testCase.config_options = kLimiterConfig3D;
  testCase.InitConfig();
  testCase.InitGeometry();
  REQUIRE(testCase.geometry->GetnDim() == 3);
  auto& g = *testCase.geometry;
  auto& c = *testCase.config;

  SECTION("3D, 1 scalar") { RunLimiterComparison<3>(g, c, kind, 1, 0, 0, 0.0, seed + 1); }
  SECTION("3D, 5 variables, padded field") { RunLimiterComparison<3>(g, c, kind, 5, 2, 0, 0.0, seed + 2); }
  SECTION("3D, 5 variables, kappa 1/3") { RunLimiterComparison<3>(g, c, kind, 5, 0, 0, 1.0 / 3.0, seed + 3); }
  SECTION("3D, 7 variables") { RunLimiterComparison<3>(g, c, kind, 7, 0, 0, -1.0, seed + 4); }
  SECTION("3D, 3 variables in a gradient that stores 5") { RunLimiterComparison<3>(g, c, kind, 3, 3, 2, 0.5, seed + 5); }
  SECTION("3D, after the GPU gradient") { RunChainedComparison<3>(g, c, kind, 5, seed + 6, 1); }
  SECTION("3D, after the GPU gradient (aliased)") { RunChainedComparison<3>(g, c, kind, 5, seed + 7, 0); }
}

void RunAllSections2D(LIMITER kind, unsigned seed) {
  UnitQuadTestCase testCase;
  testCase.config_options = kLimiterConfig2D;
  testCase.InitConfig();
  testCase.InitGeometry();
  REQUIRE(testCase.geometry->GetnDim() == 2);
  auto& g = *testCase.geometry;
  auto& c = *testCase.config;

  SECTION("2D, 1 scalar") { RunLimiterComparison<2>(g, c, kind, 1, 0, 0, 0.0, seed + 11); }
  SECTION("2D, 4 variables, padded field") { RunLimiterComparison<2>(g, c, kind, 4, 3, 0, 0.0, seed + 12); }
  SECTION("2D, 4 variables, kappa 1/3") { RunLimiterComparison<2>(g, c, kind, 4, 0, 0, 1.0 / 3.0, seed + 13); }
  SECTION("2D, 6 variables") { RunLimiterComparison<2>(g, c, kind, 6, 0, 0, -1.0, seed + 14); }
  SECTION("2D, 3 variables in a gradient that stores 4") { RunLimiterComparison<2>(g, c, kind, 3, 5, 1, 0.5, seed + 15); }
  SECTION("2D, after the GPU gradient") { RunChainedComparison<2>(g, c, kind, 4, seed + 16, 1); }
  SECTION("2D, after the GPU gradient (aliased)") { RunChainedComparison<2>(g, c, kind, 4, seed + 17, 0); }
}

}  // namespace

TEST_CASE("Barth-Jespersen limiter on the GPU matches the CPU (3D)", "[CUDA]") {
  RunAllSections3D(LIMITER::BARTH_JESPERSEN, 100);
}
TEST_CASE("Barth-Jespersen limiter on the GPU matches the CPU (2D)", "[CUDA]") {
  RunAllSections2D(LIMITER::BARTH_JESPERSEN, 100);
}
TEST_CASE("Venkatakrishnan limiter on the GPU matches the CPU (3D)", "[CUDA]") {
  RunAllSections3D(LIMITER::VENKATAKRISHNAN, 200);
}
TEST_CASE("Venkatakrishnan limiter on the GPU matches the CPU (2D)", "[CUDA]") {
  RunAllSections2D(LIMITER::VENKATAKRISHNAN, 200);
}
TEST_CASE("Nishikawa R3 limiter on the GPU matches the CPU (3D)", "[CUDA]") {
  RunAllSections3D(LIMITER::NISHIKAWA_R3, 300);
}
TEST_CASE("Nishikawa R3 limiter on the GPU matches the CPU (2D)", "[CUDA]") {
  RunAllSections2D(LIMITER::NISHIKAWA_R3, 300);
}
TEST_CASE("Nishikawa R4 limiter on the GPU matches the CPU (3D)", "[CUDA]") {
  RunAllSections3D(LIMITER::NISHIKAWA_R4, 400);
}
TEST_CASE("Nishikawa R4 limiter on the GPU matches the CPU (2D)", "[CUDA]") {
  RunAllSections2D(LIMITER::NISHIKAWA_R4, 400);
}
TEST_CASE("Nishikawa R5 limiter on the GPU matches the CPU (3D)", "[CUDA]") {
  RunAllSections3D(LIMITER::NISHIKAWA_R5, 500);
}
TEST_CASE("Nishikawa R5 limiter on the GPU matches the CPU (2D)", "[CUDA]") {
  RunAllSections2D(LIMITER::NISHIKAWA_R5, 500);
}
TEST_CASE("Venkatakrishnan-Wang limiter on the GPU matches the CPU (3D)", "[CUDA]") {
  RunAllSections3D(LIMITER::VENKATAKRISHNAN_WANG, 600);
}
TEST_CASE("Venkatakrishnan-Wang limiter on the GPU matches the CPU (2D)", "[CUDA]") {
  RunAllSections2D(LIMITER::VENKATAKRISHNAN_WANG, 600);
}

TEST_CASE("Global minimum and maximum on the GPU are exact", "[CUDA]") {
  UnitQuadTestCase testCase;
  testCase.config_options = kLimiterConfig3D;
  testCase.InitConfig();
  testCase.InitGeometry();
  auto& g = *testCase.geometry;
  auto& c = *testCase.config;

  /*--- The number of points is not a multiple of the block size and spans several blocks. ---*/
  SECTION("1 variable") { RunMinMaxCheck(g, c, 1, 0, 1); }
  SECTION("5 variables, offset 2") { RunMinMaxCheck(g, c, 5, 2, 2); }
  SECTION("16 variables") { RunMinMaxCheck(g, c, 16, 0, 3); }
}

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
