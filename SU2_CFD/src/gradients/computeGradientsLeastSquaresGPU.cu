/*!
 * @file computeGradientsLeastSquaresGPU.cu
 * @brief CUDA kernel of the Least-Squares gradient (accumulation of R and the right-hand side, and solution).
 * @note The per-point function is the one of the CPU implementation (SU2_CFD/include/gradients/leastSquaresPoint.hpp).
 *       Unlike the Green-Gauss kernels this one is NOT bit-identical to the CPU: the agreed policy is a relative
 *       difference of at most 1e-13, checked by the unit test UnitTests/SU2_CFD/gradients_gpu.cpp (observed: at most
 *       about 8e-16, a few ulp; the unweighted 3D case on the uniform test grid happens to be bit-identical).
 *       Reason: the host build is -O3 -march=native -ffast-math (GCC), which rewrites the solve. In the disassembly of
 *       the 2D solve the host replaces sqrt(x)*sqrt(x) by x (r22*r22 in the S matrix becomes the argument of the
 *       square root) and evaluates two of the divisions by detR2 as one packed division, and its contraction of the
 *       products in the accumulation is not guaranteed to match nvcc's. This file is compiled with FMA contraction on
 *       (-fmad=true, see SU2_CFD/src/meson.build) and IEEE division and square root (nvcc default), which gives the
 *       operations as written. Reproducing the host rewrites would mean changing the shared source to match one
 *       compiler's choices, which is fragile, so the tolerance is kept instead. If the host flags or compiler change,
 *       rerun the unit test.
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

#include "../../include/gradients/computeGradientsLeastSquaresGPU.hpp"

#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include "../../../Common/include/parallelization/mpi_structure.hpp"

#include "../../include/gradients/greenGaussPoint.hpp"
#include "../../include/gradients/leastSquaresPoint.hpp"

namespace ls_gpu {
namespace {

constexpr int BLOCK_SIZE = 128;

/*--- One thread per domain point, the body is the function shared with the CPU. ---*/
template <size_t nDim>
__global__ void LeastSquaresKernel(CGeometryDevice::View geom, bool weighted, const su2double* field, size_t fieldLd,
                                   su2double* grad, size_t nVarStore, size_t varBegin, size_t varEnd,
                                   size_t nPointDomain) {
  const size_t iPoint = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (iPoint >= nPointDomain) return;

  const gg::FieldView<su2double> fieldView{field, fieldLd};
  gg::GradView<su2double, nDim> gradView{grad, nVarStore};
  ls::LocalRMatrix<nDim> rMatrix;
  ls::leastSquaresPoint<nDim, gg::NoPreacc>(iPoint, geom, weighted, fieldView, varBegin, varEnd, gradView, rMatrix);
}

template <size_t nDim>
void Launch(const CGeometryDevice& geometry, bool weighted, const su2double* field, size_t fieldLd, su2double* grad,
            size_t nVarStore, size_t varBegin, size_t varEnd) {
  const size_t nPointDomain = geometry.GetnPointDomain();
  if (nPointDomain > 0) {
    const unsigned blocks = static_cast<unsigned>((nPointDomain + BLOCK_SIZE - 1) / BLOCK_SIZE);
    LeastSquaresKernel<nDim><<<blocks, BLOCK_SIZE>>>(geometry.GetGeomView(), weighted, field, fieldLd, grad,
                                                     nVarStore, varBegin, varEnd, nPointDomain);
    gpuErrChk(cudaGetLastError());
  }
  gpuErrChk(cudaDeviceSynchronize());
}

}  // namespace

void ComputeLeastSquaresDomain(const CGeometryDevice& geometry, size_t nDim, bool weighted, const su2double* field,
                               size_t fieldLd, su2double* grad, size_t nVarStore, size_t varBegin, size_t varEnd) {
  if (nDim == 2) {
    Launch<2>(geometry, weighted, field, fieldLd, grad, nVarStore, varBegin, varEnd);
  } else if (nDim == 3) {
    Launch<3>(geometry, weighted, field, fieldLd, grad, nVarStore, varBegin, varEnd);
  } else {
    SU2_MPI::Error("The least-squares gradient on the GPU supports only 2 or 3 dimensions.", CURRENT_FUNCTION);
  }
}

}  // namespace ls_gpu

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
