/*!
 * \file CSysMatrixCommsGPU.cu
 * \brief Kernels and launchers for the device-side halo exchange of CSysVector.
 * \author P. Gomes
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

/*--- Must come first, see the file for why. ---*/
#include "../../include/basic_types/codi_host_only.hpp"

#include "../../include/code_config.hpp"
#include "../../include/linear_algebra/CSysMatrixCommsGPU.hpp"
#include "../../include/linear_algebra/GPUComms.cuh"

/*--- The host logic (MPI, CGeometry) is in CSysMatrix.cpp, nvcc only sees raw pointers here. This avoids pulling
 * CGeometry/CConfig and the whole MPI machinery through nvcc. ---*/

namespace {

constexpr int COMMS_THREADS_PER_BLOCK = 256;

/*!
 * \brief One thread per (point,variable) entry: buf[k*nVar+iVar] = x[idx[k]*nVar+iVar], coalesced on the buffer.
 */
template <class T>
__global__ void PackKernel(const T* __restrict__ x, const unsigned long* __restrict__ idx, T* __restrict__ buf,
                           size_t nEntries, size_t nVar) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nEntries) return;
  const size_t k = i / nVar;
  const size_t iVar = i - k * nVar;
  buf[i] = x[idx[k] * nVar + iVar];
}

/*!
 * \brief Inverse of PackKernel. With ADD the entries are accumulated (transposed communication).
 * \note No atomics: within one message the points are unique, so one launch has no write conflicts. In the
 *       transposed case the same point can be in several messages, those are launched one after the other on the
 *       same stream, which serializes them and also keeps the summation order deterministic (an atomicAdd
 *       would not, and double atomicAdd needs sm_60+).
 */
template <class T, bool ADD>
__global__ void UnpackKernel(T* __restrict__ x, const unsigned long* __restrict__ idx, const T* __restrict__ buf,
                             size_t nEntries, size_t nVar) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nEntries) return;
  const size_t k = i / nVar;
  const size_t iVar = i - k * nVar;
  T& dst = x[idx[k] * nVar + iVar];
  if constexpr (ADD) {
    dst += buf[i];
  } else {
    dst = buf[i];
  }
}

inline unsigned int NumBlocks(size_t n) { return static_cast<unsigned int>((n + COMMS_THREADS_PER_BLOCK - 1) / COMMS_THREADS_PER_BLOCK); }

}  // namespace

namespace CSysMatrixCommsGPU {

void* Malloc(size_t bytes) {
  void* ptr = nullptr;
  gpuErrChk(cudaMalloc(&ptr, bytes));
  return ptr;
}

void Free(void* ptr) noexcept {
  /*--- Errors are ignored on purpose, the CUDA runtime may already be shut down when static destructors run. ---*/
  if (ptr) (void)cudaFree(ptr);
}

void CopyToDevice(void* dst, const void* src, size_t bytes) { gpuErrChk(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)); }

void CopyToHost(void* dst, const void* src, size_t bytes) { gpuErrChk(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost)); }

void Synchronize() { gpuErrChk(cudaStreamSynchronize(0)); }

template <class T>
void Pack(const T* x, const unsigned long* idx, T* buf, size_t nPoint, size_t nVar) {
  const size_t n = nPoint * nVar;
  if (n == 0) return;
  PackKernel<T><<<NumBlocks(n), COMMS_THREADS_PER_BLOCK>>>(x, idx, buf, n, nVar);
  gpuErrChk(cudaPeekAtLastError());
}

template <class T>
void Unpack(T* x, const unsigned long* idx, const T* buf, size_t nPoint, size_t nVar, bool add) {
  const size_t n = nPoint * nVar;
  if (n == 0) return;
  if (add) {
    UnpackKernel<T, true><<<NumBlocks(n), COMMS_THREADS_PER_BLOCK>>>(x, idx, buf, n, nVar);
  } else {
    UnpackKernel<T, false><<<NumBlocks(n), COMMS_THREADS_PER_BLOCK>>>(x, idx, buf, n, nVar);
  }
  gpuErrChk(cudaPeekAtLastError());
}

/*--- Explicit instantiations, same scalar types as the other GPU sources. ---*/

#define INSTANTIATE_COMMS_KERNELS(T)                                                                    \
  template void Pack<T>(const T*, const unsigned long*, T*, size_t, size_t);                            \
  template void Unpack<T>(T*, const unsigned long*, const T*, size_t, size_t, bool)

INSTANTIATE_COMMS_KERNELS(su2mixedfloat);

#if defined(USE_MIXED_PRECISION) && !defined(USE_SINGLE_PRECISION)
INSTANTIATE_COMMS_KERNELS(passivedouble);
#endif

#undef INSTANTIATE_COMMS_KERNELS

}  // namespace CSysMatrixCommsGPU
