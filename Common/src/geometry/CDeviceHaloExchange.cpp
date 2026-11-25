/*!
 * @file CDeviceHaloExchange.cpp
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

#include "../../include/geometry/CDeviceHaloExchange.hpp"

#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <algorithm>
#include <exception>
#include <string>
#include <type_traits>

#include "../../include/geometry/CGeometry.hpp"

static_assert(std::is_same<su2double, double>::value, "The device halo exchange works on plain doubles.");

CDeviceHaloExchange::CDeviceHaloExchange(const CGeometry& geometry)
    : nPoint(geometry.GetnPoint()), nPointDomain(geometry.GetnPointDomain()) {
  const int nSend = geometry.nP2PSend, nRecv = geometry.nP2PRecv;

  sendOffset.assign(1, 0);
  recvOffset.assign(1, 0);
  if (nSend > 0) {
    sendOffset.assign(geometry.nPoint_P2PSend, geometry.nPoint_P2PSend + nSend + 1);
    sendRank.assign(geometry.Neighbors_P2PSend, geometry.Neighbors_P2PSend + nSend);
  }
  if (nRecv > 0) {
    recvOffset.assign(geometry.nPoint_P2PRecv, geometry.nPoint_P2PRecv + nRecv + 1);
    recvRank.assign(geometry.Neighbors_P2PRecv, geometry.Neighbors_P2PRecv + nRecv);
  }
  const size_t nSendPoint = sendOffset.back(), nRecvPoint = recvOffset.back();
  if (nSendPoint > 0) hostSendIdx.assign(geometry.Local_Point_P2PSend, geometry.Local_Point_P2PSend + nSendPoint);
  if (nRecvPoint > 0) hostRecvIdx.assign(geometry.Local_Point_P2PRecv, geometry.Local_Point_P2PRecv + nRecvPoint);

  /*--- The exchange of a device array replaces the CPU one only if every halo row is received (exactly once) and
   * only domain rows are sent, which is the case for the non-periodic grids (periodic halos are not received from
   * an owner). Otherwise Exchange refuses to run. ---*/
  complete = (nRecvPoint == nPoint - nPointDomain);
  std::vector<char> seen(nPoint, 0);
  for (const auto iPoint : hostRecvIdx) {
    if (iPoint < nPointDomain || iPoint >= nPoint || seen[iPoint]) {
      complete = false;
      break;
    }
    seen[iPoint] = 1;
  }
  for (const auto iPoint : hostSendIdx) {
    if (iPoint >= nPointDomain) complete = false;
  }

  try {
    if (nSendPoint > 0) {
      sendIdx.Attach(hostSendIdx.data(), nSendPoint, "CDeviceHaloExchange::sendIdx");
      sendIdx.ToDevice();
      bytes += nSendPoint * sizeof(unsigned long);
    }
    if (nRecvPoint > 0) {
      recvIdx.Attach(hostRecvIdx.data(), nRecvPoint, "CDeviceHaloExchange::recvIdx");
      recvIdx.ToDevice();
      bytes += nRecvPoint * sizeof(unsigned long);
    }
  } catch (const std::exception& e) {
    SU2_MPI::Error(std::string("Could not copy the halo exchange lists to the GPU: ") + e.what(), CURRENT_FUNCTION);
  }

  reqSend.resize(std::max(nSend, 1));
  reqRecv.resize(std::max(nRecv, 1));
}

void CDeviceHaloExchange::EnsureCapacity(CDeviceMirror<double>& mirror, std::vector<double>& host, size_t n,
                                         const char* name) {
  if (n <= host.size() && (mirror.IsAttached() || n == 0)) return;
  /*--- Detach before the vector reallocates, the page-locked range must not outlive the memory. ---*/
  mirror.Detach();
  host.assign(std::max(n, host.size()), 0.0);
  try {
    mirror.Attach(host.data(), host.size(), name);
  } catch (const std::exception& e) {
    SU2_MPI::Error(std::string("Could not allocate the halo exchange buffers on the GPU: ") + e.what(),
                   CURRENT_FUNCTION);
  }
}

void CDeviceHaloExchange::Exchange(double* data, size_t pitch, size_t width) {
  if (!complete) {
    SU2_MPI::Error(
        "The halo points of this grid are not exactly the points received from the neighbour ranks (periodic "
        "boundaries?), the halo exchange on the GPU is not available.",
        CURRENT_FUNCTION);
  }
  if (width > pitch) SU2_MPI::Error("Invalid row layout in the halo exchange on the GPU.", CURRENT_FUNCTION);

  const size_t nSendPoint = sendOffset.back(), nRecvPoint = recvOffset.back();
  const int nSend = static_cast<int>(sendRank.size()), nRecv = static_cast<int>(recvRank.size());
  if ((nSend == 0 && nRecv == 0) || width == 0) return;

  EnsureCapacity(sendBuf, hostSendBuf, nSendPoint * width, "CDeviceHaloExchange::sendBuf");
  EnsureCapacity(recvBuf, hostRecvBuf, nRecvPoint * width, "CDeviceHaloExchange::recvBuf");

#ifdef HAVE_MPI
  /*--- Same neighbours, counts and tags as CGeometry::PostP2PRecvs and PostP2PSends. The recvs are posted first. ---*/
  for (int iRecv = 0; iRecv < nRecv; ++iRecv) {
    const int count = static_cast<int>(width * (recvOffset[iRecv + 1] - recvOffset[iRecv]));
    SU2_MPI::Irecv(hostRecvBuf.data() + width * recvOffset[iRecv], count, MPI_DOUBLE, recvRank[iRecv],
                   recvRank[iRecv] + 1, SU2_MPI::GetComm(), &reqRecv[iRecv]);
  }
#endif

  /*--- Pack all the messages (the order of the points is that of the CPU), one transfer to the host. ---*/
  DeviceHaloGPU::PackRows(data, pitch, sendIdx.DeviceData(), sendBuf.DeviceData(), nSendPoint, width);
  if (nSendPoint > 0) {
    gpuErrChk(cudaMemcpy(hostSendBuf.data(), sendBuf.DeviceData(), nSendPoint * width * sizeof(double),
                         cudaMemcpyDeviceToHost));
    bytesDown += nSendPoint * width * sizeof(double);
    ++nDown;
  }

#ifdef HAVE_MPI
  for (int iSend = 0; iSend < nSend; ++iSend) {
    const int count = static_cast<int>(width * (sendOffset[iSend + 1] - sendOffset[iSend]));
    SU2_MPI::Isend(hostSendBuf.data() + width * sendOffset[iSend], count, MPI_DOUBLE, sendRank[iSend],
                   SU2_MPI::GetRank() + 1, SU2_MPI::GetComm(), &reqSend[iSend]);
  }

  if (nRecv > 0) SU2_MPI::Waitall(nRecv, reqRecv.data(), MPI_STATUSES_IGNORE);
#endif

  if (nRecvPoint > 0) {
    gpuErrChk(cudaMemcpy(recvBuf.DeviceData(), hostRecvBuf.data(), nRecvPoint * width * sizeof(double),
                         cudaMemcpyHostToDevice));
    bytesUp += nRecvPoint * width * sizeof(double);
    ++nUp;
    DeviceHaloGPU::UnpackRows(data, pitch, recvIdx.DeviceData(), recvBuf.DeviceData(), nRecvPoint, width);
  }

#ifdef HAVE_MPI
  /*--- The sends read the host buffer, it must not be reused before they are done. ---*/
  if (nSend > 0) SU2_MPI::Waitall(nSend, reqSend.data(), MPI_STATUSES_IGNORE);
#endif
}

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
