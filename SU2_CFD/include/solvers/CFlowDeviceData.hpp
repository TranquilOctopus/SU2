/*!
 * @file CFlowDeviceData.hpp
 * @brief Device (GPU) mirrors of the per-point data of one finite-volume flow solver (CUDA_OFFLOAD).
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

#include "../../../Common/include/code_config.hpp"

/*--- Only exists in builds with CUDA that are not AD builds, the header is empty otherwise. ---*/
#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <cstddef>
#include <string>
#include <vector>

#include "../../../Common/include/containers/CDeviceMirror.hpp"

/*!
 * @brief The device mirrors of one flow solver (finest grid), owned by CFVMFlowSolverBase and created lazily by the
 *        first offloaded call.
 * @details The mirrors borrow (and page-lock) host arrays owned by the CVariable of the solver, so this object has
 *          to be destroyed before them. As a member of the solver it is, the solver's destructor body runs before
 *          the base class CSolver releases the variables.
 *
 *          Who owns what is current (the rules every user of the mirrors has to follow):
 *          - primitives: the CPU writes them (SetPrimitive_Variables, and a few other places, all of which call
 *            PrimitivesModified, see CFVMFlowSolverBase::MarkPrimitivesHostModified), the GPU only reads them. The
 *            upload is lazy: the first consumer after a CPU write calls primitives.ToDevice(), the others find
 *            the mirror current and copy nothing.
 *          - gradient[0], gradient[1]: written by the GPU kernels (all the rows are mirrored). The domain rows are
 *            computed by the kernels and the halo rows are received from the owner ranks by the device halo exchange
 *            (CDeviceHaloExchange), so the device copy is the final gradient. Only the variables computed by the
 *            call are downloaded (all rows), the variables that a call does not compute are neither uploaded nor
 *            downloaded, they only live on the host (see ComputeGradientsGreenGauss).
 *          - limiter: written by the limiter kernels (all the rows: the domain rows are computed and the halo rows
 *            received by the device halo exchange, like the gradients), downloaded once after the exchange. It is
 *            read only by the host. solutionMin, solutionMax, rMatrix: reserved, never attached (the limiter on the
 *            GPU does not store the min/max over the neighbors: on the CPU they are only read by the periodic
 *            communications, and periodic markers are not supported by the offload).
 */
struct CFlowDeviceData {
  CDeviceMirror<su2double> primitives;  /*!< @brief All points, nPrimVar columns. Written by the CPU. */
  CDeviceMirror<su2double> gradient[2]; /*!< @brief [0] viscous gradient, [1] reconstruction gradient (all rows). */

  CDeviceMirror<su2double> limiter;     /*!< @brief Limiter of the primitive variables (all rows). Written by the GPU. */

  /*--- Reserved (min/max over the neighbors for the limiters, least-squares matrix), not used yet. ---*/
  CDeviceMirror<su2double> solutionMin;
  CDeviceMirror<su2double> solutionMax;
  CDeviceMirror<su2double> rMatrix;

  std::vector<char> useMarker; /*!< @brief Scratch: markers that contribute to the Green-Gauss gradient. */

  /*! @brief Declare that the CPU wrote the primitives (the device copy is stale until the next upload). */
  void PrimitivesModified() { primitives.HostModified(); }

  /*!
   * @brief Attach a mirror to a host array, first banking the counters of what it did before so that the totals
   *        never go down when an array is re-attached (it moved).
   */
  void Attach(CDeviceMirror<su2double>& mirror, su2double* host, size_t n, const std::string& name) {
    Bank(mirror);
    mirror.Attach(host, n, name);
  }

  /*!
   * @brief Add the staging copies of a halo exchange (device to host for the send buffer, host to device for the
   *        received data) to the totals.
   */
  void AddHaloTraffic(size_t bytesUp, size_t bytesDown, size_t numUp, size_t numDown) {
    haloUp += bytesUp;
    haloDown += bytesDown;
    haloNumUp += numUp;
    haloNumDown += numDown;
  }

  /*!
   * @brief Add a transfer that does not go through a mirror (e.g. the small result of a reduction) to the totals.
   */
  void AddTransfer(size_t bytesUp, size_t bytesDown, size_t numUp, size_t numDown) {
    otherUp += bytesUp;
    otherDown += bytesDown;
    otherNumUp += numUp;
    otherNumDown += numDown;
  }

  /*! @brief All bytes copied host to device, and device to host, by all the mirrors and halo exchanges since the creation. */
  size_t BytesUploaded() const { return bankedUp + haloUp + otherUp + SumOf(&CDeviceMirror<su2double>::BytesUploaded); }
  size_t BytesDownloaded() const {
    return bankedDown + haloDown + otherDown + SumOf(&CDeviceMirror<su2double>::BytesDownloaded);
  }

  /*! @brief All host to device and device to host transfers since the creation. */
  size_t NumUploads() const { return bankedNumUp + haloNumUp + otherNumUp + SumOf(&CDeviceMirror<su2double>::NumUploads); }
  size_t NumDownloads() const {
    return bankedNumDown + haloNumDown + otherNumDown + SumOf(&CDeviceMirror<su2double>::NumDownloads);
  }

  /*! @brief The part of the totals that is halo staging traffic. */
  size_t HaloBytesUploaded() const { return haloUp; }
  size_t HaloBytesDownloaded() const { return haloDown; }

 private:
  size_t bankedUp = 0, bankedDown = 0, bankedNumUp = 0, bankedNumDown = 0;
  size_t haloUp = 0, haloDown = 0, haloNumUp = 0, haloNumDown = 0;
  size_t otherUp = 0, otherDown = 0, otherNumUp = 0, otherNumDown = 0;

  void Bank(const CDeviceMirror<su2double>& m) {
    bankedUp += m.BytesUploaded();
    bankedDown += m.BytesDownloaded();
    bankedNumUp += m.NumUploads();
    bankedNumDown += m.NumDownloads();
  }

  size_t SumOf(size_t (CDeviceMirror<su2double>::*what)() const) const {
    return (primitives.*what)() + (gradient[0].*what)() + (gradient[1].*what)() + (limiter.*what)() +
           (solutionMin.*what)() + (solutionMax.*what)() + (rMatrix.*what)();
  }
};

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
