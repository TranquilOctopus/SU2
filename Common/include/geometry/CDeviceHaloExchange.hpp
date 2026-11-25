/*!
 * @file CDeviceHaloExchange.hpp
 * @brief MPI halo exchange of a row-major per-point array that lives on the GPU.
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

#include "../code_config.hpp"

/*--- Only exists in builds with CUDA that are not AD builds (see code_config.hpp), the header is empty otherwise. ---*/
#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <cstddef>
#include <vector>

#include "../containers/CDeviceMirror.hpp"
#include "../parallelization/mpi_structure.hpp"

class CGeometry;

/*!
 * @namespace DeviceHaloGPU
 * @brief CUDA-free interface to the pack/unpack kernels of the halo exchange (CDeviceHaloExchangeGPU.cu).
 *        Everything works on raw device pointers and is enqueued on the default stream.
 */
namespace DeviceHaloGPU {

/*!
 * @brief Gather buf[k*width+j] = x[idx[k]*pitch+j] for k in [0,nPoint) and j in [0,width).
 * @param[in] x - Device array, one row of length pitch per point.
 * @param[in] pitch - Row length of x.
 * @param[in] idx - Device array with the rows to gather.
 * @param[out] buf - Device buffer, nPoint*width entries.
 * @param[in] nPoint - Number of rows to gather.
 * @param[in] width - Number of entries taken from the start of each row, at most pitch.
 */
void PackRows(const double* x, size_t pitch, const unsigned long* idx, double* buf, size_t nPoint, size_t width);

/*! @brief Inverse of PackRows: x[idx[k]*pitch+j] = buf[k*width+j]. The indices must be unique. */
void UnpackRows(double* x, size_t pitch, const unsigned long* idx, const double* buf, size_t nPoint, size_t width);

}  // namespace DeviceHaloGPU

/*!
 * @class CDeviceHaloExchange
 * @brief Fills the halo rows of a row-major per-point array on the device with the values of the owner ranks.
 * @details Does the same as the point-to-point communication of CSolver::InitiateComms/CompleteComms (same
 * neighbours, message sizes, tags and order of the points in the messages), starting from device data: the rows of
 * the send points are packed by a kernel, copied to a page-locked host buffer, exchanged with non-blocking MPI calls
 * (the MPI is not CUDA-aware, so the data is staged through the host), copied back, and unpacked into the halo rows
 * by a kernel. The index lists (send and receive points) are copied to the device once, when the object is built.
 *
 * Only the first width entries of every row are exchanged, which is what the CPU does for a quantity that has fewer
 * entries per point than the container (the gradient of nPrimVarGrad variables in a container of more variables).
 * It can be used for any per-point quantity of the same structure, e.g. the limiter (PRIMITIVE_LIMITER: pitch =
 * width = nPrimVarGrad).
 *
 * The exchange has to be called by a single thread per rank (inside BEGIN/END_SU2_OMP_SAFE_GLOBAL_ACCESS, it contains
 * no OpenMP directives) and by all the ranks, it is not thread safe. Only the non-periodic exchange of the finest
 * grid is supported: the halo points must be exactly the points that are received (IsComplete()).
 */
class CDeviceHaloExchange {
  size_t nPoint = 0, nPointDomain = 0;
  bool complete = false;

  /*--- Copies of the point-to-point structure of the geometry (offsets in points, one more than messages). ---*/
  std::vector<int> sendOffset, recvOffset, sendRank, recvRank;
  std::vector<unsigned long> hostSendIdx, hostRecvIdx;
  /*--- Page-locked staging buffers, the mirrors page-lock them and own the device side (the buffers of the pack and
   * unpack kernels). They are declared after the vectors they borrow, so they are destroyed first. ---*/
  std::vector<double> hostSendBuf, hostRecvBuf;
  CDeviceMirror<unsigned long> sendIdx, recvIdx;
  CDeviceMirror<double> sendBuf, recvBuf;
  std::vector<SU2_MPI::Request> reqSend, reqRecv;

  size_t bytes = 0;
  size_t bytesUp = 0, bytesDown = 0, nUp = 0, nDown = 0;

  void EnsureCapacity(CDeviceMirror<double>& mirror, std::vector<double>& host, size_t n, const char* name);

 public:
  /*!
   * @brief Copy the point-to-point structure of the geometry and upload the index lists.
   * @note The geometry must be the finest grid with its point-to-point structure set up. Errors end the run.
   */
  explicit CDeviceHaloExchange(const CGeometry& geometry);

  CDeviceHaloExchange(const CDeviceHaloExchange&) = delete;
  CDeviceHaloExchange& operator=(const CDeviceHaloExchange&) = delete;

  /*!
   * @brief Whether the received points are exactly the halo points [nPointDomain, nPoint), each once, and the sent
   *        points are domain points. Otherwise (e.g. periodic halos) the exchange cannot replace the CPU one.
   */
  bool IsComplete() const { return complete; }

  /*! @brief Total size in bytes of the device index lists. */
  size_t GetBytes() const { return bytes; }

  /*!
   * @brief Exchange (collective, blocking).
   * @param[in,out] data - Device array with nPoint rows of length pitch. The halo rows are overwritten.
   * @param[in] pitch - Row length.
   * @param[in] width - Number of entries from the start of each row that are exchanged (at most pitch).
   */
  void Exchange(double* data, size_t pitch, size_t width);

  /*! @brief Bytes (and number of copies) moved by the staging copies since the creation, host to device and back. */
  size_t BytesUploaded() const { return bytesUp; }
  size_t BytesDownloaded() const { return bytesDown; }
  size_t NumUploads() const { return nUp; }
  size_t NumDownloads() const { return nDown; }
};

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
