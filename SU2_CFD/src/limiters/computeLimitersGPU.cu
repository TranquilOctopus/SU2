/*!
 * @file computeLimitersGPU.cu
 * @brief CUDA kernels of the slope limiters of the primitive variables (the computation of one point and the global
 *        minimum and maximum of a field for Venkatakrishnan-Wang).
 * @note The per-point function is the one of the CPU implementation (SU2_CFD/include/limiters/limiterPoint.hpp), and
 *       so are the limiter functions (CLimiterDetails.hpp). Like the least-squares gradient, and unlike the
 *       Green-Gauss kernels, the result is NOT guaranteed to be bit-identical to the CPU: the agreed policy is a
 *       relative difference of at most 1e-13, checked by the unit test UnitTests/SU2_CFD/limiters_gpu.cpp
 *       (observed on the test grids, 2D and 3D, 1 to 7 variables, all six kinds: 10 to 40 percent of the entries
 *       differ in bits, by at most 1.3e-15 absolute, i.e. a few ulp of values in [0,1]; the largest relative
 *       difference is 2.7e-14, for a limiter of about 1e-2 whose absolute error is 3e-16, all the others are below
 *       3e-15; Venkatakrishnan-Wang differs least, 1 ulp in 0.5 to 5 percent of the entries). Reason: the host build is
 *       -O3 -march=native -ffast-math (GCC), which is free to rewrite the limiter functions (reciprocal of a
 *       repeated denominator, pow(x, n) with integer n expanded into products with its own association, the choice
 *       of the products that are contracted into FMAs), while this file is compiled with FMA contraction on
 *       (-fmad=true, see SU2_CFD/src/meson.build) and IEEE division, which gives the operations as written. The
 *       global minimum and maximum (Venkatakrishnan-Wang) are exact operations and have no such dependence on the
 *       order, they are bit-identical to the CPU.
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

#include "../../include/limiters/computeLimitersGPU.hpp"

#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <algorithm>
#include <vector>

#include "../../../Common/include/parallelization/mpi_structure.hpp"

#include "../../include/limiters/limiterPoint.hpp"

namespace lim_gpu {
namespace {

constexpr int BLOCK_SIZE = 128;

/*--- Gradient of all the points as the limiter function reads it: gradient[iPoint][iVar] is the pointer to the nDim
 *    components, with the layout (point, variable, dimension) of the gradient mirror. ---*/
template <size_t nDim>
struct GradientRows {
  const su2double* ptr;
  size_t ld; /*!< Row length of one point (variables stored times nDim). */

  struct Point {
    const su2double* row;
    SU2_HOST_DEVICE const su2double* operator[](size_t iVar) const { return row + iVar * nDim; }
  };
  SU2_HOST_DEVICE Point operator[](size_t iPoint) const { return Point{ptr + iPoint * ld}; }
};

/*--- Writable (iPoint, iVar) matrix. ---*/
struct MatrixWriter {
  su2double* ptr;
  size_t ld;
  SU2_HOST_DEVICE su2double& operator()(size_t iPoint, size_t iVar) const { return ptr[iPoint * ld + iVar]; }
};

/*--- The min/max of the field over the neighbors of the point only live in the registers of the thread. ---*/
struct LocalVariables {
  su2double v[lim::MAXNVAR];
  SU2_HOST_DEVICE su2double& operator()(size_t, size_t iVar) { return v[iVar]; }
};

/*--- Nothing to pass to the geometricFactor of the kinds that run on the GPU (it is 1). ---*/
struct NoGeometry {};

/*--- One thread per domain point, the body is the function shared with the CPU. ---*/
template <size_t nDim, LIMITER Kind>
__global__ void LimiterKernel(CGeometryDevice::View geom, CLimiterDetails<Kind> details, su2double umusclKappa,
                              const su2double* field, size_t fieldLd, const su2double* grad, size_t gradLd,
                              su2double* limiter, size_t limiterLd, size_t varBegin, size_t varEnd,
                              size_t nPointDomain) {
  const size_t iPoint = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (iPoint >= nPointDomain) return;

  const gg::FieldView<su2double> fieldView{field, fieldLd};
  const GradientRows<nDim> gradView{grad, gradLd};
  MatrixWriter limiterView{limiter, limiterLd};
  LocalVariables fieldMin, fieldMax;
  NoGeometry geoContext;

  lim::limiterPoint<nDim, gg::NoPreacc>(iPoint, geom, details, geoContext, false, nullptr, 0, varBegin, varEnd,
                                        umusclKappa, fieldView, gradView, fieldMin, fieldMax, limiterView);
}

template <size_t nDim, LIMITER Kind>
void Launch(const CGeometryDevice& geometry, const CLimiterDetails<Kind>& details, su2double umusclKappa,
            const su2double* field, size_t fieldLd, const su2double* grad, size_t gradLd, su2double* limiter,
            size_t limiterLd, size_t varBegin, size_t varEnd) {
  const size_t nPointDomain = geometry.GetnPointDomain();
  if (nPointDomain > 0) {
    const unsigned blocks = static_cast<unsigned>((nPointDomain + BLOCK_SIZE - 1) / BLOCK_SIZE);
    LimiterKernel<nDim, Kind><<<blocks, BLOCK_SIZE>>>(geometry.GetGeomView(), details, umusclKappa, field, fieldLd,
                                                      grad, gradLd, limiter, limiterLd, varBegin, varEnd,
                                                      nPointDomain);
    gpuErrChk(cudaGetLastError());
  }
  gpuErrChk(cudaDeviceSynchronize());
}

constexpr int REDUCE_BLOCK = 256;
constexpr size_t REDUCE_MAX_BLOCKS = 128;

/*--- Each block reduces, for every variable, its share of the points (grid-stride loop) to one minimum and one
 *    maximum, by a tree reduction in shared memory with a fixed pairing. The partial results are stored in order
 *    of block and variable. ---*/
__global__ void MinMaxPartialKernel(const su2double* field, size_t fieldLd, size_t nPointDomain, size_t varBegin,
                                    size_t nVar, su2double largeNum, su2double* partial) {
  __shared__ su2double sharedMin[REDUCE_BLOCK], sharedMax[REDUCE_BLOCK];

  for (size_t iVar = 0; iVar < nVar; ++iVar) {
    su2double lo = largeNum, hi = -largeNum;
    for (size_t iPoint = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; iPoint < nPointDomain;
         iPoint += static_cast<size_t>(gridDim.x) * blockDim.x) {
      const su2double x = field[iPoint * fieldLd + varBegin + iVar];
      lo = min(lo, x);
      hi = max(hi, x);
    }
    sharedMin[threadIdx.x] = lo;
    sharedMax[threadIdx.x] = hi;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        sharedMin[threadIdx.x] = min(sharedMin[threadIdx.x], sharedMin[threadIdx.x + stride]);
        sharedMax[threadIdx.x] = max(sharedMax[threadIdx.x], sharedMax[threadIdx.x + stride]);
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      partial[(static_cast<size_t>(blockIdx.x) * nVar + iVar) * 2] = sharedMin[0];
      partial[(static_cast<size_t>(blockIdx.x) * nVar + iVar) * 2 + 1] = sharedMax[0];
    }
    __syncthreads();
  }
}

/*--- Second pass: one thread per variable combines the partial results in the order of the blocks. ---*/
__global__ void MinMaxFinalKernel(const su2double* partial, unsigned nBlocks, size_t nVar, su2double* result) {
  const size_t iVar = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (iVar >= nVar) return;

  su2double lo = partial[iVar * 2], hi = partial[iVar * 2 + 1];
  for (unsigned iBlock = 1; iBlock < nBlocks; ++iBlock) {
    lo = min(lo, partial[(static_cast<size_t>(iBlock) * nVar + iVar) * 2]);
    hi = max(hi, partial[(static_cast<size_t>(iBlock) * nVar + iVar) * 2 + 1]);
  }
  result[iVar * 2] = lo;
  result[iVar * 2 + 1] = hi;
}

}  // namespace

template <LIMITER Kind>
void ComputeLimiterDomain(const CGeometryDevice& geometry, size_t nDim, const CLimiterDetails<Kind>& details,
                          su2double umusclKappa, const su2double* field, size_t fieldLd, const su2double* grad,
                          size_t gradLd, su2double* limiter, size_t limiterLd, size_t varBegin, size_t varEnd) {
  if (nDim == 2) {
    Launch<2, Kind>(geometry, details, umusclKappa, field, fieldLd, grad, gradLd, limiter, limiterLd, varBegin,
                    varEnd);
  } else if (nDim == 3) {
    Launch<3, Kind>(geometry, details, umusclKappa, field, fieldLd, grad, gradLd, limiter, limiterLd, varBegin,
                    varEnd);
  } else {
    SU2_MPI::Error("The limiters on the GPU support only 2 or 3 dimensions.", CURRENT_FUNCTION);
  }
}

/*--- One kernel template serves all the kinds, these are the kinds that run on the GPU. ---*/
#define INSTANTIATE(KIND)                                                                                    \
  template void ComputeLimiterDomain<KIND>(const CGeometryDevice&, size_t, const CLimiterDetails<KIND>&,   \
                                            su2double, const su2double*, size_t, const su2double*, size_t,  \
                                            su2double*, size_t, size_t, size_t);
INSTANTIATE(LIMITER::BARTH_JESPERSEN)
INSTANTIATE(LIMITER::VENKATAKRISHNAN)
INSTANTIATE(LIMITER::NISHIKAWA_R3)
INSTANTIATE(LIMITER::NISHIKAWA_R4)
INSTANTIATE(LIMITER::NISHIKAWA_R5)
INSTANTIATE(LIMITER::VENKATAKRISHNAN_WANG)
#undef INSTANTIATE

void ReduceFieldMinMax(const CGeometryDevice& geometry, const su2double* field, size_t fieldLd, size_t varBegin,
                       size_t varEnd, su2double largeNum, su2double* minimum, su2double* maximum) {
  const size_t nVar = varEnd - varBegin;
  const size_t nPointDomain = geometry.GetnPointDomain();
  if (nVar == 0) return;

  const unsigned nBlocks = static_cast<unsigned>(
      std::max<size_t>(1, std::min<size_t>(REDUCE_MAX_BLOCKS, (nPointDomain + REDUCE_BLOCK - 1) / REDUCE_BLOCK)));

  su2double *partial = nullptr, *result = nullptr;
  gpuErrChk(cudaMalloc(&partial, static_cast<size_t>(nBlocks) * nVar * 2 * sizeof(su2double)));
  gpuErrChk(cudaMalloc(&result, nVar * 2 * sizeof(su2double)));

  MinMaxPartialKernel<<<nBlocks, REDUCE_BLOCK>>>(field, fieldLd, nPointDomain, varBegin, nVar, largeNum, partial);
  gpuErrChk(cudaGetLastError());
  MinMaxFinalKernel<<<static_cast<unsigned>((nVar + 31) / 32), 32>>>(partial, nBlocks, nVar, result);
  gpuErrChk(cudaGetLastError());

  std::vector<su2double> host(nVar * 2);
  gpuErrChk(cudaMemcpy(host.data(), result, nVar * 2 * sizeof(su2double), cudaMemcpyDeviceToHost));
  gpuErrChk(cudaFree(partial));
  gpuErrChk(cudaFree(result));

  for (size_t iVar = 0; iVar < nVar; ++iVar) {
    minimum[iVar] = host[iVar * 2];
    maximum[iVar] = host[iVar * 2 + 1];
  }
}

}  // namespace lim_gpu

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
