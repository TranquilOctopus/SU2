/*!
 * \file CDeviceMirror_kernels.cu
 * \brief The tiny kernel used by the CDeviceMirror unit tests.
 * \version 8.6.0 "Harrier"
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

#include "CDeviceMirror_kernels.hpp"

#include "../../../Common/include/code_config.hpp"
#include "../../../Common/include/linear_algebra/GPUComms.cuh"

#ifdef SU2_ENABLE_CUDA_KERNELS

/*--- A function shared by host and device code, which is what SU2_HOST_DEVICE is for. ---*/
SU2_HOST_DEVICE inline double Twice(double x) { return 2 * x; }

namespace {
__global__ void DoubleKernel(double* data, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) data[i] = Twice(data[i]);
}
}  // namespace

void CDeviceMirrorTestDouble(double* devPtr, size_t n) {
  constexpr int blockSize = 128;
  const int nBlocks = int((n + blockSize - 1) / blockSize);
  DoubleKernel<<<nBlocks, blockSize>>>(devPtr, n);
  gpuErrChk(cudaPeekAtLastError());
  gpuErrChk(cudaDeviceSynchronize());
}

double CDeviceMirrorTestTwiceOnHost(double x) { return Twice(x); }

#endif
