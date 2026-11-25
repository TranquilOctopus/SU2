/*!
 * \file computeLimitersGPU.hpp
 * \brief Slope limiters of the primitive variables on the GPU (CUDA_OFFLOAD= (GRADIENTS, LIMITERS)).
 * \author P. Gomes (original implementation), adapted for the GPU offload
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

#include "../../../Common/include/code_config.hpp"

/*--- Only exists in builds with CUDA that are not AD builds, the header is empty otherwise. ---*/
#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "../../../Common/include/containers/CDeviceMirror.hpp"
#include "../../../Common/include/geometry/CGeometryDevice.hpp"
#include "../solvers/CFlowDeviceData.hpp"

/*--- The declarations of the kernels are also seen by nvcc, the host infrastructure is not. ---*/
#ifndef __CUDACC__
#include "../../../Common/include/CConfig.hpp"
#include "../../../Common/include/geometry/CGeometry.hpp"
#include "../../../Common/include/parallelization/mpi_structure.hpp"
#include "../../../Common/include/parallelization/omp_structure.hpp"
#endif

#include "CLimiterDetails.hpp"

namespace lim_gpu {

/*!
 * @brief Compute the limiter of all the variables of the domain points on the device (the same as the loop over the
 *        points of computeLimiters_impl does on the CPU for periodic = false, one thread per point, the body is
 *        shared with the CPU). Defined for the kinds Barth-Jespersen, Venkatakrishnan, Nishikawa R3, R4 and R5
 *        and Venkatakrishnan-Wang.
 * @note Synchronous. The halo rows of the limiter are not touched. The min/max of the field over the neighbors
 *       (fieldMin, fieldMax of the CPU function) are only used inside the point computation, they are not stored.
 * @param[in] geometry - Device copy of the geometry.
 * @param[in] nDim - Number of dimensions, 2 or 3.
 * @param[in] details - Limiter parameters, after "preprocess".
 * @param[in] umusclKappa - Blending parameter for U-MUSCL reconstruction.
 * @param[in] field - Device array, row-major, one row of length fieldLd per point (all points, incl. halos).
 * @param[in] fieldLd - Row length of the field.
 * @param[in] grad - Device array, row-major (point, variable, dimension), with gradLd/nDim variables per point.
 * @param[in] gradLd - Row length of the gradient (the number of variables stored per point times nDim).
 * @param[out] limiter - Device array, row-major, one row of length limiterLd per point.
 * @param[in] limiterLd - Row length of the limiter.
 * @param[in] varBegin - First variable for which the limiter is computed.
 * @param[in] varEnd - End (exclusive) of the range of variables.
 */
template <LIMITER Kind>
void ComputeLimiterDomain(const CGeometryDevice& geometry, size_t nDim, const CLimiterDetails<Kind>& details,
                          su2double umusclKappa, const su2double* field, size_t fieldLd, const su2double* grad,
                          size_t gradLd, su2double* limiter, size_t limiterLd, size_t varBegin, size_t varEnd);

/*!
 * @brief Minimum and maximum of each variable of a field over the domain points of the rank, computed on the device.
 * @details Deterministic: the min and max of a block are found by a tree reduction in a fixed order, and the results
 *          of the blocks are combined by a second kernel (one thread per variable) in the order of the blocks, so
 *          there are no atomics and the result is the same in every run (min and max are exact, so it is also the
 *          result of any other order, in particular that of the CPU).
 * @note Synchronous. The result (two values per variable) is copied to the host.
 * @param[in] geometry - Device copy of the geometry.
 * @param[in] field - Device array, row-major, one row of length fieldLd per point.
 * @param[in] fieldLd - Row length of the field.
 * @param[in] varBegin - First variable.
 * @param[in] varEnd - End (exclusive) of the range of variables.
 * @param[in] largeNum - Value of the minimum if there are no points (the maximum is -largeNum).
 * @param[out] minimum - Host array, varEnd - varBegin values.
 * @param[out] maximum - Host array, varEnd - varBegin values.
 */
void ReduceFieldMinMax(const CGeometryDevice& geometry, const su2double* field, size_t fieldLd, size_t varBegin,
                       size_t varEnd, su2double largeNum, su2double* minimum, su2double* maximum);

#ifndef __CUDACC__

/*! @brief Whether the limiter is computed on the GPU (see ComputeLimiters). */
inline bool IsSupported(LIMITER kind) {
  switch (kind) {
    case LIMITER::BARTH_JESPERSEN:
    case LIMITER::VENKATAKRISHNAN:
    case LIMITER::NISHIKAWA_R3:
    case LIMITER::NISHIKAWA_R4:
    case LIMITER::NISHIKAWA_R5:
    case LIMITER::VENKATAKRISHNAN_WANG:
      return true;
    default:
      return false;
  }
}

/*! @brief Name of the limiter for messages. */
inline const char* Name(LIMITER kind) {
  switch (kind) {
    case LIMITER::BARTH_JESPERSEN: return "Barth-Jespersen";
    case LIMITER::VENKATAKRISHNAN: return "Venkatakrishnan";
    case LIMITER::NISHIKAWA_R3: return "Nishikawa R3";
    case LIMITER::NISHIKAWA_R4: return "Nishikawa R4";
    case LIMITER::NISHIKAWA_R5: return "Nishikawa R5";
    case LIMITER::VENKATAKRISHNAN_WANG: return "Venkatakrishnan-Wang";
    default: return "unsupported";
  }
}

namespace detail {

/*! @brief The limiter of one kind: "preprocess" on the host, then the kernel. */
template <LIMITER Kind>
void RunKind(CFlowDeviceData& data, CGeometry& geometry, const CGeometryDevice& device, const CConfig& config,
             su2double umusclKappa, const su2double* gradient, size_t nPrimVar, size_t nVarStore, size_t nVarLim,
             size_t limiterLd) {
  const size_t nDim = device.GetnDim();
  const size_t varBegin = 0, varEnd = nVarLim;

  CLimiterDetails<Kind> details;
  const su2double* prim = data.primitives.DeviceRead();

  if constexpr (Kind == LIMITER::VENKATAKRISHNAN_WANG) {
    /*--- Global minimum and maximum of each variable: per rank on the device, then the reduction between ranks of
     * the CPU implementation (see CLimiterDetails<VENKATAKRISHNAN_WANG>::preprocess). ---*/

    const su2double largeNum = 0.1 * std::numeric_limits<passivedouble>::max();

    std::vector<su2double> rankMin(nVarLim), rankMax(nVarLim);
    ReduceFieldMinMax(device, prim, nPrimVar, varBegin, varEnd, largeNum, rankMin.data(), rankMax.data());
    data.AddTransfer(0, 2 * nVarLim * sizeof(su2double), 0, 1);

    su2activevector localMin(varEnd), localMax(varEnd), globalMin(varEnd), globalMax(varEnd);
    localMin = largeNum;
    localMax = -largeNum;
    for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
      localMin(iVar) = rankMin[iVar - varBegin];
      localMax(iVar) = rankMax[iVar - varBegin];
    }
    SU2_MPI::Allreduce(localMin.data(), globalMin.data(), varEnd, MPI_DOUBLE, MPI_MIN, SU2_MPI::GetComm());
    SU2_MPI::Allreduce(localMax.data(), globalMax.data(), varEnd, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

    details.setEps2(config, varBegin, varEnd, globalMin, globalMax);
  } else {
    struct NoField {} noField;
    size_t begin = varBegin, end = varEnd;
    details.preprocess(geometry, config, begin, end, noField);
  }

  ComputeLimiterDomain<Kind>(device, nDim, details, umusclKappa, prim, nPrimVar, gradient,
                             nVarStore * nDim, data.limiter.DeviceWrite(), limiterLd, varBegin, varEnd);
}

}  // namespace detail

/*!
 * @brief Run the limiter of the primitive variables on the GPU: the limiter of every variable at the domain points
 *        (for Venkatakrishnan-Wang after the global min/max), the halo exchange on the device, and one download.
 * @details Called by a single thread of every rank (no OpenMP synchronization inside, MPI calls inside). Periodic
 *          markers are not supported (the offload configuration rejects them), so the minimum and maximum of the
 *          field over the neighbors, which on the CPU are written to CVariable::Solution_Min/Max and only read by the
 *          periodic communications, are neither stored nor downloaded, and the periodic corrections do not apply.
 *
 *          Data flow: the primitives and the reconstruction gradient (the gradient that the CPU limiter reads) are
 *          read from their device mirrors. After the gradient offload both are current on the device, and nothing is
 *          copied (ToDevice is a no-op when the device copy is current). If the gradient mirror does not exist yet,
 *          or the host copy is the newer one (the gradient was written by the CPU), it is created and uploaded here
 *          (all rows), so the function is also correct when the gradient did not come from the GPU. The primitives
 *          are uploaded lazily in the same way (see CFlowDeviceData).
 *
 *          The limiter of the domain points is computed by the kernel, the halo rows are received from the owner
 *          ranks by the device halo exchange (the same values as the CPU communication PRIMITIVE_LIMITER: the first
 *          nVarLim variables of every point), and the result is downloaded once, all rows.
 * @param[in,out] data - Mirrors, attached on the first call (and again if the host arrays moved).
 * @param[in] geometry - Finest grid (for the host part of the limiter parameters).
 * @param[in] device - Device copy of the geometry.
 * @param[in] config - Configuration.
 * @param[in] kind - Kind of limiter, must be supported (IsSupported).
 * @param[in] umusclKappa - Blending parameter for U-MUSCL reconstruction.
 * @param[in] hostPrimitives - Host primitives, nPoint rows of length nPrimVar.
 * @param[in] nPrimVar - Row length of the primitives.
 * @param[in] hostGradient - Host reconstruction gradient, nPoint rows of nVarStore x nDim.
 * @param[in] nVarStore - Number of variables per point stored in the gradient (its row length / nDim).
 * @param[in] nVarLim - Number of variables for which the limiter is computed (the first ones).
 * @param[out] hostLimiter - Host limiter, nPoint rows of length limiterLd.
 * @param[in] limiterLd - Row length of the limiter (at least nVarLim).
 * @param[out] firstCall - Set to true if the limiter mirror was created by this call (for messages).
 */
inline void ComputeLimiters(CFlowDeviceData& data, CGeometry& geometry, const CGeometryDevice& device,
                            const CConfig& config, LIMITER kind, su2double umusclKappa,
                            const su2double* hostPrimitives, size_t nPrimVar, const su2double* hostGradient,
                            size_t nVarStore, size_t nVarLim, su2double* hostLimiter, size_t limiterLd,
                            bool& firstCall) {
  const size_t nPoint = device.GetnPoint(), nDim = device.GetnDim();

  if (nVarLim > LIMITER_MAX_NVAR) {
    SU2_MPI::Error("Number of variables is too large for the limiters on the GPU, increase LIMITER_MAX_NVAR.",
                   CURRENT_FUNCTION);
  }

  auto& prim = data.primitives;
  /*--- The gradient that the CPU limiter reads is the reconstruction gradient, which the variables of the flow solvers
   * alias to the viscous gradient when the same method is used for both (CFlowVariable): then there is one host
   * array, and the device copy that the gradient offload keeps current is the mirror of the viscous gradient. ---*/
  const bool aliased = data.gradient[0].IsAttached() && data.gradient[0].HostData() == hostGradient;
  auto& grad = aliased ? data.gradient[0] : data.gradient[1];
  auto& limiter = data.limiter;
  firstCall = false;

  if (!prim.IsAttached() || prim.HostData() != hostPrimitives) {
    data.Attach(prim, const_cast<su2double*>(hostPrimitives), nPoint * nPrimVar, "limiter primitives");
  }
  if (!grad.IsAttached() || grad.HostData() != hostGradient) {
    data.Attach(grad, const_cast<su2double*>(hostGradient), nPoint * nVarStore * nDim, "limiter reconstruction gradient");
  }
  /*--- All the points: the halo rows are filled on the device by the halo exchange below. ---*/
  if (!limiter.IsAttached() || limiter.HostData() != hostLimiter) {
    data.Attach(limiter, hostLimiter, nPoint * limiterLd, "limiter");
    firstCall = true;
  }

  /*--- No-ops if the device copies are current (the usual case), see the description. ---*/
  prim.ToDevice();
  grad.ToDevice();
  const su2double* gradDevice = grad.DeviceRead();

  switch (kind) {
    case LIMITER::BARTH_JESPERSEN:
      detail::RunKind<LIMITER::BARTH_JESPERSEN>(data, geometry, device, config, umusclKappa, gradDevice, nPrimVar,
                                                nVarStore, nVarLim, limiterLd);
      break;
    case LIMITER::VENKATAKRISHNAN:
      detail::RunKind<LIMITER::VENKATAKRISHNAN>(data, geometry, device, config, umusclKappa, gradDevice, nPrimVar,
                                                nVarStore, nVarLim, limiterLd);
      break;
    case LIMITER::NISHIKAWA_R3:
      detail::RunKind<LIMITER::NISHIKAWA_R3>(data, geometry, device, config, umusclKappa, gradDevice, nPrimVar,
                                             nVarStore, nVarLim, limiterLd);
      break;
    case LIMITER::NISHIKAWA_R4:
      detail::RunKind<LIMITER::NISHIKAWA_R4>(data, geometry, device, config, umusclKappa, gradDevice, nPrimVar,
                                             nVarStore, nVarLim, limiterLd);
      break;
    case LIMITER::NISHIKAWA_R5:
      detail::RunKind<LIMITER::NISHIKAWA_R5>(data, geometry, device, config, umusclKappa, gradDevice, nPrimVar,
                                             nVarStore, nVarLim, limiterLd);
      break;
    case LIMITER::VENKATAKRISHNAN_WANG:
      detail::RunKind<LIMITER::VENKATAKRISHNAN_WANG>(data, geometry, device, config, umusclKappa, gradDevice,
                                                     nPrimVar, nVarStore, nVarLim, limiterLd);
      break;
    default:
      SU2_MPI::Error("This limiter is not available on the GPU.", CURRENT_FUNCTION);
  }

  /*--- The halo rows from the owners, the same values as the CPU communication gives. ---*/
  auto& halo = device.GetHalo();
  const size_t up0 = halo.BytesUploaded(), down0 = halo.BytesDownloaded();
  const size_t nUp0 = halo.NumUploads(), nDown0 = halo.NumDownloads();
  halo.Exchange(limiter.DeviceData(), limiterLd, nVarLim);
  data.AddHaloTraffic(halo.BytesUploaded() - up0, halo.BytesDownloaded() - down0, halo.NumUploads() - nUp0,
                      halo.NumDownloads() - nDown0);

  limiter.DeviceModified();
  if (limiterLd == nVarLim) {
    limiter.ToHost();
  } else {
    limiter.ToHostRowPrefix(nVarLim, limiterLd);
  }
}

#endif  // __CUDACC__

}  // namespace lim_gpu

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
