/*!
 * @file CSysGPU_tests.cpp
 * @brief Unit tests that compare the CUDA linear algebra with the CPU path: CSysVector expressions and
 *        reductions, the block sparse matrix-vector product, the Jacobi, LU-SGS and ILU preconditioners,
 *        and an FGMRES solve. Every test computes the same quantity with ENABLE_CUDA= NO and
 *        ENABLE_CUDA= YES on the same data and compares the results element-wise.
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
#include "../../UnitQuadTestCase.hpp"
#include "../../../Common/include/linear_algebra/CSysMatrix.hpp"
#include "../../../Common/include/linear_algebra/CSysSolve.hpp"
#include "../../../Common/include/linear_algebra/CPreconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef SU2_ENABLE_CUDA_KERNELS

namespace {

using TestScalar = su2mixedfloat;
using TestVec = CSysVector<TestScalar>;
using TestMat = CSysMatrix<TestScalar>;

/*--- Largest nVar used here, the block scratch arrays are sized for it. ---*/
constexpr unsigned short kMaxTestVar = 5;

/*--- Relative tolerance for vector expressions, dot products (scaled by the sum of absolute terms, since
 * the device reduction uses atomics and reorders the sum) and the matrix-vector product. The kernels
 * are double precision, the only differences are reordering and possible fused multiply-adds, which stay
 * at a few ulp. ---*/
constexpr double kRelTol = 1e-12;

/*--- The ILU factorization on the device is computed by a fixed number of colored sweeps. With enough
 * sweeps it converges to the exact ILU(0) factors, which are unique for this pattern, so the comparison is
 * against the CPU factorization with a looser tolerance (the fixed point converges geometrically for a
 * diagonally dominant matrix; the rate here is well below 0.1 per sweep). ---*/
constexpr double kIluTol = 1e-10;
constexpr unsigned long kIluSweeps = 50;

/*! @brief Enables device vector expressions for its lifetime, always switches them off again. */
struct DeviceExpressions {
  explicit DeviceExpressions(bool on) { VecExpr::SetUseDeviceExpressions(on); }
  ~DeviceExpressions() { VecExpr::SetUseDeviceExpressions(false); }
};

/*! @brief Builds a config from the default unit test options plus one option line per entry. */
std::unique_ptr<CConfig> MakeConfig(std::initializer_list<std::string> lines) {
  UnitQuadTestCase base;
  for (const auto& line : lines) base.AddOption(line);
  std::stringstream ss(base.config_options);
  cout.rdbuf(nullptr);
  auto config = std::unique_ptr<CConfig>(new CConfig(ss, SU2_COMPONENT::SU2_CFD, false));
  cout.rdbuf(base.orig_buf);
  return config;
}

/*! @brief The 5x5x5 box mesh of the unit tests, used as the sparsity pattern of the matrices. */
struct Mesh {
  UnitQuadTestCase testCase;
  Mesh() {
    testCase.InitConfig();
    testCase.InitGeometry();
  }
  CGeometry* geo() { return testCase.geometry.get(); }
};

std::unique_ptr<TestMat> MakeMatrix(CGeometry* geo, const CConfig* config, unsigned short nVar) {
  auto mat = std::unique_ptr<TestMat>(new TestMat());
  mat->Initialize(geo->GetnPoint(), geo->GetnPointDomain(), nVar, nVar, true, geo, config);
  return mat;
}

/*! @brief Fills a diagonally dominant block matrix on the edge pattern of the mesh, deterministically from seed.
 * The diagonal blocks are 10 * identity plus noise, the off-diagonal blocks are small noise, so every
 * preconditioner is well conditioned. Calling it twice with the same seed gives identical host values. */
void FillDiagonallyDominant(TestMat& mat, CGeometry* geo, unsigned short nVar, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> noise(-0.2, 0.2);

  su2double blk[kMaxTestVar][kMaxTestVar];
  su2double blk2[kMaxTestVar][kMaxTestVar];
  su2double zero[kMaxTestVar][kMaxTestVar] = {};

  auto randomBlock = [&](su2double (&out)[kMaxTestVar][kMaxTestVar], double diagonal) {
    for (unsigned short i = 0; i < kMaxTestVar; ++i)
      for (unsigned short j = 0; j < kMaxTestVar; ++j) out[i][j] = (i == j ? diagonal : 0.0) + noise(gen);
  };

  mat.SetValZero();
  for (unsigned long iPoint = 0; iPoint < geo->GetnPointDomain(); ++iPoint) {
    randomBlock(blk, 10.0);
    mat.SetBlock2Diag(iPoint, blk);
  }
  for (unsigned long iEdge = 0; iEdge < geo->GetnEdge(); ++iEdge) {
    const auto iPoint = geo->edges->GetNode(iEdge, 0);
    const auto jPoint = geo->edges->GetNode(iEdge, 1);
    randomBlock(blk, 0.0);
    randomBlock(blk2, 0.0);
    mat.SetBlocks(iEdge, iPoint, jPoint, zero, blk, blk2, zero);
  }
}

/*! @brief Deterministic host data in [-1, 1]. */
std::vector<double> RandomHost(unsigned long n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> values(n);
  for (auto& v : values) v = dist(gen);
  return values;
}

TestVec MakeVec(const std::vector<double>& host) {
  TestVec v(static_cast<unsigned long>(host.size()), 0.0);
  for (unsigned long i = 0; i < host.size(); ++i) v[i] = host[i];
  return v;
}

/*! @brief Compares two vectors on their first n entries, the worst relative difference (absolute below 1) must
 * stay within rtol. */
void CheckClose(const TestVec& ref, const TestVec& got, unsigned long n, double rtol) {
  double worstRel = 0.0;
  unsigned long worstIdx = 0;
  for (unsigned long i = 0; i < n; ++i) {
    const double r = SU2_TYPE::GetValue(ref[i]);
    const double g = SU2_TYPE::GetValue(got[i]);
    const double rel = std::fabs(g - r) / std::max(1.0, std::fabs(r));
    if (rel > worstRel) {
      worstRel = rel;
      worstIdx = i;
    }
  }
  INFO("worst relative difference " << worstRel << " at entry " << worstIdx << " of " << n);
  CHECK(worstRel <= rtol);
}

/*! @brief The expression shapes that the device vector kernels are instantiated for (CSysVectorGPU.cu). */
constexpr int kNumExpressions = 13;

void ApplyExpression(int kind, TestVec& R, const TestVec& A, const TestVec& B, const TestVec& C) {
  const TestScalar s1 = 0.7, s2 = -0.3, s3 = 1.25;
  switch (kind) {
    case 0:
      R = s1 * A;
      break;
    case 1:
      R = A * s1;
      break;
    case 2:
      R = A / s1;
      break;
    case 3:
      R = A * s1 + B * s2;
      break;
    case 4:
      R = A * s1 + B * s2 + C * s3;
      break;
    case 5:
      R = A - B;
      break;
    case 6:
      R = s1 * A + B;
      break;
    case 7:
      R = s1 * (A - s2 * B);
      break;
    case 8:
      R = s1 * (A - s2 * B) + C;
      break;
    case 9:
      R = A;
      break;
    case 10:
      R = s1;
      break;
    case 11:
      R = B;
      R -= s1 * A;
      break;
    default:
      R = B;
      R += A * s2;
      break;
  }
}

}  // namespace

TEST_CASE("CSysVector expressions on the device match the CPU", "[LinearAlgebra][CUDA]") {
  const unsigned long sizes[] = {1, 7, 1000, 100003};
  for (const auto n : sizes) {
    const auto A = MakeVec(RandomHost(n, 11));
    const auto B = MakeVec(RandomHost(n, 22));
    const auto C = MakeVec(RandomHost(n, 33));

    for (int kind = 0; kind < kNumExpressions; ++kind) {
      INFO("expression " << kind << ", size " << n);

      TestVec Rcpu(n, 0.0), Rgpu(n, 0.0);
      ApplyExpression(kind, Rcpu, A, B, C);

      /*--- The inputs cross the bus once, the result comes back once. ---*/
      A.HtDTransfer();
      B.HtDTransfer();
      C.HtDTransfer();
      {
        DeviceExpressions on(true);
        ApplyExpression(kind, Rgpu, A, B, C);
      }
      Rgpu.DtHTransfer();

      CheckClose(Rcpu, Rgpu, n, kRelTol);
    }
  }
}

TEST_CASE("CSysVector dot, norm and multiDot on the device match the CPU", "[LinearAlgebra][CUDA]") {
  const unsigned long sizes[] = {1, 7, 1000, 100003};
  for (const auto n : sizes) {
    INFO("size " << n);
    const auto ha = RandomHost(n, 44);
    const auto hb = RandomHost(n, 55);
    const auto A = MakeVec(ha);
    const auto B = MakeVec(hb);

    /*--- Scale of the rounding error: the sum of the absolute terms. ---*/
    double termScale = 0.0;
    for (unsigned long i = 0; i < n; ++i) termScale += std::fabs(ha[i] * hb[i]);

    const double cpuDot = SU2_TYPE::GetValue(A.dot(B));
    const double cpuNorm = SU2_TYPE::GetValue(A.norm());

    A.HtDTransfer();
    B.HtDTransfer();
    double gpuDot = 0.0, gpuNorm = 0.0;
    {
      DeviceExpressions on(true);
      gpuDot = SU2_TYPE::GetValue(A.dot(B));
      gpuNorm = SU2_TYPE::GetValue(A.norm());
    }

    CHECK(std::fabs(gpuDot - cpuDot) <= kRelTol * termScale);
    CHECK(std::fabs(gpuNorm - cpuNorm) <= kRelTol * cpuNorm);

    /*--- Tall matrices V (3 columns) and W (2 columns), one more random set for each. ---*/
    std::vector<TestVec> V, W;
    std::vector<std::vector<double>> hV, hW;
    for (unsigned seed = 100; seed < 103; ++seed) {
      hV.push_back(RandomHost(n, seed));
      V.push_back(MakeVec(hV.back()));
    }
    for (unsigned seed = 200; seed < 202; ++seed) {
      hW.push_back(RandomHost(n, seed));
      W.push_back(MakeVec(hW.back()));
    }

    const su2matrix<TestScalar> cpuM = TestVec::multiDot(V, 0, 3, W, 2);

    for (const auto& v : V) v.HtDTransfer();
    for (const auto& w : W) w.HtDTransfer();
    su2matrix<TestScalar> gpuM;
    {
      DeviceExpressions on(true);
      gpuM = TestVec::multiDot(V, 0, 3, W, 2);
    }

    REQUIRE(cpuM.rows() == 3);
    REQUIRE(cpuM.cols() == 2);
    REQUIRE(gpuM.rows() == 3);
    REQUIRE(gpuM.cols() == 2);
    for (unsigned long i = 0; i < 3; ++i)
      for (unsigned long j = 0; j < 2; ++j) {
        double scale = 0.0;
        for (unsigned long k = 0; k < n; ++k) scale += std::fabs(hV[i][k] * hW[j][k]);
        const double r = SU2_TYPE::GetValue(cpuM(i, j));
        const double g = SU2_TYPE::GetValue(gpuM(i, j));
        INFO("multiDot entry " << i << "," << j);
        CHECK(std::fabs(g - r) <= kRelTol * scale);
      }
  }
}

static void CheckMatVec(unsigned short nVar) {
  INFO("nVar " << nVar);
  Mesh mesh;
  CGeometry* geo = mesh.geo();
  auto cpuConfig = MakeConfig({"ENABLE_CUDA= NO", "LINEAR_SOLVER_PREC= ILU"});
  auto gpuConfig = MakeConfig({"ENABLE_CUDA= YES", "LINEAR_SOLVER_PREC= ILU"});

  auto matCpu = MakeMatrix(geo, cpuConfig.get(), nVar);
  auto matGpu = MakeMatrix(geo, gpuConfig.get(), nVar);
  FillDiagonallyDominant(*matCpu, geo, nVar, 1001);
  FillDiagonallyDominant(*matGpu, geo, nVar, 1001);

  const unsigned long nPoint = geo->GetnPoint();
  const unsigned long nPointDomain = geo->GetnPointDomain();
  const unsigned long nElm = nPoint * nVar;

  TestVec x(nPoint, nPointDomain, nVar, 0.0);
  const auto hx = RandomHost(nElm, 2002);
  for (unsigned long i = 0; i < nElm; ++i) x[i] = hx[i];

  TestVec yCpu(nPoint, nPointDomain, nVar, 0.0), yGpu(nPoint, nPointDomain, nVar, 0.0);
  matCpu->MatrixVectorProduct(x, yCpu, geo, cpuConfig.get());

  x.HtDTransfer();
  matGpu->HtDTransfer();
  matGpu->MatrixVectorProduct(x, yGpu, geo, gpuConfig.get());
  yGpu.DtHTransfer();

  CheckClose(yCpu, yGpu, nPointDomain * nVar, kRelTol);
}

TEST_CASE("Block sparse matrix-vector product on the device matches the CPU", "[LinearAlgebra][CUDA]") {
  CheckMatVec(1);
  CheckMatVec(4);
  CheckMatVec(5);
}

static void CheckPreconditioner(const std::string& prec, const std::string& gpuOption, unsigned short nVar,
                                double rtol) {
  INFO("preconditioner " << prec << ", nVar " << nVar);
  Mesh mesh;
  CGeometry* geo = mesh.geo();
  auto cpuConfig = MakeConfig({"ENABLE_CUDA= NO", prec});
  auto gpuConfig = MakeConfig({"ENABLE_CUDA= YES", prec, gpuOption});

  auto matCpu = MakeMatrix(geo, cpuConfig.get(), nVar);
  auto matGpu = MakeMatrix(geo, gpuConfig.get(), nVar);
  FillDiagonallyDominant(*matCpu, geo, nVar, 3003);
  FillDiagonallyDominant(*matGpu, geo, nVar, 3003);

  const unsigned long nPoint = geo->GetnPoint();
  const unsigned long nPointDomain = geo->GetnPointDomain();
  const unsigned long nElm = nPoint * nVar;

  TestVec x(nPoint, nPointDomain, nVar, 0.0);
  const auto hx = RandomHost(nElm, 4004);
  for (unsigned long i = 0; i < nElm; ++i) x[i] = hx[i];

  const auto kind = static_cast<ENUM_LINEAR_SOLVER_PREC>(cpuConfig->GetKind_Linear_Solver_Prec());
  TestVec yCpu(nPoint, nPointDomain, nVar, 0.0), yGpu(nPoint, nPointDomain, nVar, 0.0);

  auto precCpu = std::unique_ptr<CPreconditioner<TestScalar>>(
      CPreconditioner<TestScalar>::Create(kind, *matCpu, geo, cpuConfig.get()));
  precCpu->Build();
  (*precCpu)(x, yCpu);

  /*--- The device preconditioner is built from the device copy of the matrix, which the solver uploads
   * before the build (see CSysMatrixVectorProduct). ---*/
  matGpu->HtDTransfer();
  auto precGpu = std::unique_ptr<CPreconditioner<TestScalar>>(
      CPreconditioner<TestScalar>::Create(kind, *matGpu, geo, gpuConfig.get()));
  precGpu->Build();

  /*--- As in CSysSolve, the device preconditioner runs with the vectors on the device and the device
   * expression switch on. Without the switch the wrappers take the host path and never touch the device. ---*/
  x.HtDTransfer();
  {
    DeviceExpressions on(true);
    (*precGpu)(x, yGpu);
  }
  yGpu.DtHTransfer();

  CheckClose(yCpu, yGpu, nPointDomain * nVar, rtol);
}

TEST_CASE("Jacobi preconditioner on the device matches the CPU", "[LinearAlgebra][CUDA]") {
  CheckPreconditioner("LINEAR_SOLVER_PREC= JACOBI", "", 1, kRelTol);
  CheckPreconditioner("LINEAR_SOLVER_PREC= JACOBI", "", 4, kRelTol);
  CheckPreconditioner("LINEAR_SOLVER_PREC= JACOBI", "", 5, kRelTol);
}

TEST_CASE("LU-SGS preconditioner on the device matches the CPU", "[LinearAlgebra][CUDA]") {
  CheckPreconditioner("LINEAR_SOLVER_PREC= LU_SGS", "", 1, kRelTol);
  CheckPreconditioner("LINEAR_SOLVER_PREC= LU_SGS", "", 4, kRelTol);
  CheckPreconditioner("LINEAR_SOLVER_PREC= LU_SGS", "", 5, kRelTol);
}

TEST_CASE("ILU preconditioner on the device converges to the CPU ILU(0)", "[LinearAlgebra][CUDA]") {
  const std::string sweeps = "LINEAR_SOLVER_ILU_GPU_SWEEPS= " + std::to_string(kIluSweeps);
  CheckPreconditioner("LINEAR_SOLVER_PREC= ILU", sweeps, 1, kIluTol);
  CheckPreconditioner("LINEAR_SOLVER_PREC= ILU", sweeps, 4, kIluTol);
  CheckPreconditioner("LINEAR_SOLVER_PREC= ILU", sweeps, 5, kIluTol);
}

TEST_CASE("FGMRES solve on the device matches the CPU", "[LinearAlgebra][CUDA]") {
  const unsigned short nVar = 4;
  Mesh mesh;
  CGeometry* geo = mesh.geo();
  const std::string solver = "LINEAR_SOLVER= FGMRES";
  const std::string prec = "LINEAR_SOLVER_PREC= ILU";
  const std::string error = "LINEAR_SOLVER_ERROR= 1E-10";
  const std::string iter = "LINEAR_SOLVER_ITER= 500";
  const std::string sweeps = "LINEAR_SOLVER_ILU_GPU_SWEEPS= " + std::to_string(kIluSweeps);
  auto cpuConfig = MakeConfig({"ENABLE_CUDA= NO", solver, prec, error, iter});
  auto gpuConfig = MakeConfig({"ENABLE_CUDA= YES", solver, prec, error, iter, sweeps});

  auto matCpu = MakeMatrix(geo, cpuConfig.get(), nVar);
  auto matGpu = MakeMatrix(geo, gpuConfig.get(), nVar);
  FillDiagonallyDominant(*matCpu, geo, nVar, 5005);
  FillDiagonallyDominant(*matGpu, geo, nVar, 5005);

  const unsigned long nPoint = geo->GetnPoint();
  const unsigned long nPointDomain = geo->GetnPointDomain();
  const unsigned long nElm = nPoint * nVar;

  /*--- Manufactured solution, the right hand side is A times it. ---*/
  TestVec xTrue(nPoint, nPointDomain, nVar, 0.0);
  const auto hx = RandomHost(nElm, 6006);
  for (unsigned long i = 0; i < nElm; ++i) xTrue[i] = hx[i];
  TestVec rhs(nPoint, nPointDomain, nVar, 0.0);
  matCpu->MatrixVectorProduct(xTrue, rhs, geo, cpuConfig.get());

  TestVec xCpu(nPoint, nPointDomain, nVar, 0.0), xGpu(nPoint, nPointDomain, nVar, 0.0);

  CSysSolve<TestScalar> solverCpu;
  const auto iterCpu = solverCpu.Solve(*matCpu, rhs, xCpu, geo, cpuConfig.get());

  CSysSolve<TestScalar> solverGpu;
  const auto iterGpu = solverGpu.Solve(*matGpu, rhs, xGpu, geo, gpuConfig.get());

  CHECK(iterCpu < 500);
  CHECK(iterGpu < 500);

  double normTrue = 0.0;
  for (unsigned long i = 0; i < nElm; ++i) normTrue = std::max(normTrue, std::fabs(hx[i]));
  double errCpu = 0.0, errGpu = 0.0;
  for (unsigned long i = 0; i < nElm; ++i) {
    errCpu = std::max(errCpu, std::fabs(SU2_TYPE::GetValue(xCpu[i]) - hx[i]));
    errGpu = std::max(errGpu, std::fabs(SU2_TYPE::GetValue(xGpu[i]) - hx[i]));
  }
  INFO("iterations CPU " << iterCpu << ", GPU " << iterGpu);
  CHECK(errCpu <= 1e-7 * normTrue);
  CHECK(errGpu <= 1e-7 * normTrue);

  /*--- The two solutions are not bitwise equal (different reduction orders), they must agree to the
   * accuracy of the solve. ---*/
  CheckClose(xCpu, xGpu, nPointDomain * nVar, 1e-7);
}

#endif  // SU2_ENABLE_CUDA_KERNELS
