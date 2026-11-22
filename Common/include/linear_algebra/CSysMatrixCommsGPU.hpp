/*!
 * \file CSysMatrixCommsGPU.hpp
 * \brief Declarations of the device helpers used by the GPU halo exchange (CSysMatrixComms::InitiateGPU/CompleteGPU).
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

#pragma once

#include <cstddef>

/*!
 * \namespace CSysMatrixCommsGPU
 * \brief Thin, CUDA-free interface to the device side of the GPU halo exchange. Everything works on raw
 *        pointers so that this header can be included from host-only translation units (CSysMatrix.cpp), the
 *        implementation lives in CSysMatrixCommsGPU.cu. All operations are enqueued on the default stream, the
 *        same one used by the other GPU linear algebra kernels, so they are ordered with respect to them.
 */
namespace CSysMatrixCommsGPU {

/*! \brief Allocate device memory, aborts on failure. */
void* Malloc(size_t bytes);

/*! \brief Free device memory, never throws or aborts (called from static destructors at exit). */
void Free(void* ptr) noexcept;

/*! \brief Copy from host to device (blocking for the host unless the source is pinned). */
void CopyToDevice(void* dst, const void* src, size_t bytes);

/*! \brief Copy from device to host, blocking. */
void CopyToHost(void* dst, const void* src, size_t bytes);

/*! \brief Wait for all work enqueued on the default stream. */
void Synchronize();

/*!
 * \brief Gather buf[k*nVar+iVar] = x[idx[k]*nVar+iVar] for k in [0,nPoint).
 * \param[in] x      - Device vector (point-major layout).
 * \param[in] idx    - Device array with the point indices to gather.
 * \param[out] buf   - Device buffer, nPoint*nVar entries.
 * \param[in] nPoint - Number of points to gather.
 * \param[in] nVar   - Number of variables per point.
 */
template <class T>
void Pack(const T* x, const unsigned long* idx, T* buf, size_t nPoint, size_t nVar);

/*!
 * \brief Scatter x[idx[k]*nVar+iVar] (=|+=) buf[k*nVar+iVar] for k in [0,nPoint).
 * \note With add==true the indices must be unique within one call (true for the points of one message), concurrent
 *       calls for different messages are serialized by the stream, so no atomics are needed (see the .cu file).
 */
template <class T>
void Unpack(T* x, const unsigned long* idx, const T* buf, size_t nPoint, size_t nVar, bool add);

}  // namespace CSysMatrixCommsGPU
