/*!
 * \file allocation_toolbox.hpp
 * \brief Helper function and classes for memory allocation.
 *        Focus on portability across platforms.
 * \note  These are "kernel" functions, only to be used with good reason,
 *        always try to use higher level container classes.
 * \author P. Gomes, D. Kavolis
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

#pragma once

#if defined(_WIN32)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#ifdef HAVE_CUDA
#include "../linear_algebra/GPUComms.cuh"
#endif

#include <cstring>

#include <cassert>

namespace MemoryAllocation {

inline constexpr bool is_power_of_two(size_t x) { return x && !(x & (x - 1)); }

inline constexpr size_t round_up(size_t multiple, size_t x) { return ((x + multiple - 1) / multiple) * multiple; }

/*!
 * \brief Aligned memory allocation compatible across platforms.
 * \param[in] alignment, in bytes, of the memory being allocated.
 * \param[in] size, also in bytes.
 * \tparam ZeroInit, initialize memory to 0.
 * \return Pointer to memory, always use su2::aligned_free to deallocate.
 */
template <class T, bool ZeroInit = false>
inline T* aligned_alloc(size_t alignment, size_t size) noexcept {
  assert(is_power_of_two(alignment));

  if (alignment < alignof(void*)) alignment = alignof(void*);

  size = round_up(alignment, size);

  void* ptr = nullptr;

  if (size > 0) {
#if defined(__APPLE__)
    if (::posix_memalign(&ptr, alignment, size) != 0) {
      ptr = nullptr;
    }
#elif defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    ptr = ::aligned_alloc(alignment, size);
#endif
    if (ZeroInit) memset(ptr, 0, size);
  }
  return static_cast<T*>(ptr);
}

/*!
 * \brief Free memory allocated with su2::aligned_alloc.
 * \param[in] ptr, pointer to memory we want to release.
 */
template <class T>
inline void aligned_free(T* ptr) noexcept {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace MemoryAllocation

namespace GPUMemoryAllocation {
/*!
 * \brief Memory allocation for variables on the GPU.
 * \param[in] size in bytes.
 * \tparam ZeroInit, initialize memory to 0.
 * \return Pointer to memory, always use gpu_free to deallocate.
 */
template <class T, bool ZeroInit = false>
inline T* gpu_alloc(size_t size) noexcept {
  void* ptr = nullptr;

#if defined(HAVE_CUDA)
  gpuErrChk(cudaMalloc((void**)(&ptr), size));
  if (ZeroInit) gpuErrChk(cudaMemset((void*)(ptr), 0.0, size));
#else
  return 0;
#endif

  return static_cast<T*>(ptr);
}

/*!
 * \brief Free memory allocated on the GPU with gpu_alloc.
 * \param[in] ptr, pointer to memory we want to release.
 */
template <class T>
inline void gpu_free(T* ptr) noexcept {
#ifdef HAVE_CUDA
  gpuErrChk(cudaFree((void*)ptr));
#endif
}
/*!
 * \brief Memory allocation for variables on the GPU along with initialization from a source host array.
 * \param[in] size in bytes.
 * \return Pointer to memory, always use gpu_free to deallocate.
 */
template <class T>
inline T* gpu_alloc_cpy(const T* src_ptr, size_t size) noexcept {
  void* ptr = nullptr;

#ifdef HAVE_CUDA
  gpuErrChk(cudaMalloc((void**)(&ptr), size));
  gpuErrChk(cudaMemcpy((void*)(ptr), (void*)src_ptr, size, cudaMemcpyHostToDevice));
#endif

  return static_cast<T*>(ptr);
}

/*!
 * \brief Page-locked ("pinned") host memory allocation.
 * \note Unlike regular (pageable) host memory, cudaMemcpyAsync from/to a pinned buffer is
 *       actually asynchronous with respect to the host thread; from pageable memory the driver
 *       silently falls back to a synchronous staged copy. Only worth it for host buffers that
 *       are the source/destination of an async transfer meant to overlap with other host work.
 * \param[in] size in bytes.
 * \tparam ZeroInit, initialize memory to 0.
 * \return Pointer to memory, always use pinned_free to deallocate.
 */
template <class T, bool ZeroInit = false>
inline T* pinned_alloc(size_t size) noexcept {
  void* ptr = nullptr;

#if defined(HAVE_CUDA)
  gpuErrChk(cudaMallocHost((void**)(&ptr), size));
  if (ZeroInit) memset(ptr, 0, size);
#else
  return 0;
#endif

  return static_cast<T*>(ptr);
}

/*!
 * \brief Free memory allocated with pinned_alloc.
 * \param[in] ptr, pointer to memory we want to release.
 */
template <class T>
inline void pinned_free(T* ptr) noexcept {
#ifdef HAVE_CUDA
  gpuErrChk(cudaFreeHost((void*)ptr));
#endif
}

/*!
 * \brief Page-lock an existing host allocation (cudaHostRegister) so that transfers from/to it run at
 *        pinned-memory speed, without changing how it is allocated or freed.
 * \note Never fatal: if registering fails (pinned-memory limit, overlap with another registered range...)
 *       a warning is printed once per process and the caller keeps using the pageable buffer.
 *       The memory must be unregistered (pinned_unregister) before it is freed or reallocated.
 * \return True if the range is now registered.
 */
inline bool pinned_register(void* ptr, size_t size) noexcept {
#if defined(HAVE_CUDA)
  if (ptr == nullptr || size == 0) return false;
  const auto err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
  if (err == cudaSuccess) return true;
  (void)cudaGetLastError();  // clear the (non sticky) error so later checks do not trip over it
  static bool warned = false;
  if (!warned) {
    warned = true;
    std::cerr << "WARNING: could not page-lock host memory for CUDA transfers (" << cudaGetErrorString(err)
              << "), falling back to pageable (slower) copies." << std::endl;
  }
#endif
  return false;
}

/*!
 * \brief Undo pinned_register, only call for ranges that were successfully registered.
 */
inline void pinned_unregister(void* ptr) noexcept {
#if defined(HAVE_CUDA)
  if (ptr != nullptr) (void)cudaHostUnregister(ptr);
#endif
}
}  // namespace GPUMemoryAllocation
