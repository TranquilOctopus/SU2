/*!
 * @file CDeviceHaloExchangeGPU.cu
 * @brief Pack and unpack kernels of the device halo exchange (the MPI part is in CDeviceHaloExchange.cpp).
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

/*--- Must come first, see the file for why. ---*/
#include "../../include/basic_types/codi_host_only.hpp"

#include "../../include/code_config.hpp"
#include "../../include/geometry/CDeviceHaloExchange.hpp"
#include "../../include/linear_algebra/GPUComms.cuh"

#ifdef SU2_ENABLE_GEOMETRY_DEVICE

namespace {

constexpr int HALO_THREADS_PER_BLOCK = 256;

/*--- One thread per entry, coalesced on the buffer. ---*/
__global__ void PackRowsKernel(const double* __restrict__ x, size_t pitch, const unsigned long* __restrict__ idx,
                               double* __restrict__ buf, size_t nEntries, size_t width) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nEntries) return;
  const size_t k = i / width;
  const size_t j = i - k * width;
  buf[i] = x[idx[k] * pitch + j];
}

/*--- The indices are unique (checked when the lists are built), so there are no write conflicts. ---*/
__global__ void UnpackRowsKernel(double* __restrict__ x, size_t pitch, const unsigned long* __restrict__ idx,
                                 const double* __restrict__ buf, size_t nEntries, size_t width) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nEntries) return;
  const size_t k = i / width;
  const size_t j = i - k * width;
  x[idx[k] * pitch + j] = buf[i];
}

inline unsigned int NumBlocks(size_t n) {
  return static_cast<unsigned int>((n + HALO_THREADS_PER_BLOCK - 1) / HALO_THREADS_PER_BLOCK);
}

}  // namespace

namespace DeviceHaloGPU {

void PackRows(const double* x, size_t pitch, const unsigned long* idx, double* buf, size_t nPoint, size_t width) {
  const size_t n = nPoint * width;
  if (n == 0) return;
  PackRowsKernel<<<NumBlocks(n), HALO_THREADS_PER_BLOCK>>>(x, pitch, idx, buf, n, width);
  gpuErrChk(cudaPeekAtLastError());
}

void UnpackRows(double* x, size_t pitch, const unsigned long* idx, const double* buf, size_t nPoint, size_t width) {
  const size_t n = nPoint * width;
  if (n == 0) return;
  UnpackRowsKernel<<<NumBlocks(n), HALO_THREADS_PER_BLOCK>>>(x, pitch, idx, buf, n, width);
  gpuErrChk(cudaPeekAtLastError());
}

}  // namespace DeviceHaloGPU

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
