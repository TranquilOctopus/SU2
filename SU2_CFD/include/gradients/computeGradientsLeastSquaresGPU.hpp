/*!
 * \file computeGradientsLeastSquaresGPU.hpp
 * \brief Least-Squares (unweighted and inverse-distance weighted) gradient of the primitive variables on the GPU.
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
#include <vector>

#include "../../../Common/include/containers/CDeviceMirror.hpp"
#include "../../../Common/include/geometry/CGeometryDevice.hpp"
#include "../solvers/CFlowDeviceData.hpp"
#include "computeGradientsGreenGaussGPU.hpp"

namespace ls_gpu {

/*!
 * @brief Accumulate the R matrix and the right-hand side and solve the least-squares problem at all domain points
 *        on the device (the same as the first loop of computeGradientsLeastSquares does on the CPU, one thread per
 *        point, the body is shared with the CPU).
 * @note Synchronous. The R matrix of a point lives in the registers of its thread, it is not stored: on the CPU it
 *       is only read again by the periodic communications (see ComputeGradientsLeastSquares). Halo rows of the
 *       gradient are not touched.
 * @param[in] geometry - Device copy of the geometry.
 * @param[in] nDim - Number of dimensions, 2 or 3.
 * @param[in] weighted - Use inverse-distance weights.
 * @param[in] field - Device array, row-major, one row of length fieldLd per point (all points, incl. halos).
 * @param[in] fieldLd - Row length of the field.
 * @param[out] grad - Device array, row-major (point, variable, dimension), with nVarStore variables per point.
 * @param[in] nVarStore - Number of variables per point stored in the gradient.
 * @param[in] varBegin - First variable for which the gradient is computed.
 * @param[in] varEnd - End (exclusive) of the range of variables.
 */
void ComputeLeastSquaresDomain(const CGeometryDevice& geometry, size_t nDim, bool weighted, const su2double* field,
                               size_t fieldLd, su2double* grad, size_t nVarStore, size_t varBegin, size_t varEnd);

/*!
 * @brief Run the least-squares gradient of the primitive variables on the GPU: the accumulation and solution at the
 *        domain points, the symmetry and Euler wall corrections (after the solve, like on the CPU), the halo
 *        exchange of the gradient on the device and one download of the result.
 * @details Same contract as gg_gpu::ComputeGradientsGreenGauss (mirrors, lazy upload of the primitives, handling of
 *          the variables that are stored but not computed, no periodic markers, called by a single thread of every
 *          rank), see there.
 *
 *          The R matrix of the CPU (CVariable::Rmatrix) is neither computed nor downloaded: the only readers of it
 *          are the periodic communications (CSolver::CompletePeriodicComms and the setup of the second loop of
 *          computeGradientsLeastSquares), and the offload does not support periodic markers. The R matrix is
 *          rebuilt from scratch at every gradient computation, so nobody reads a stale value.
 * @param[in,out] data - Mirrors, attached on the first call (and again if the host arrays moved).
 * @param[in] geometry - Finest grid.
 * @param[in] weighted - Use inverse-distance weights.
 * @param[in] hostPrimitives - Host primitives, nPoint rows of length nPrimVar.
 * @param[in] nPrimVar - Row length of the primitives.
 * @param[in] hostGradient - Host gradient, nPoint rows of nVarStore x nDim.
 * @param[in] nVarStore - Number of variables per point stored in the gradient (its row length).
 * @param[in] nVarGrad - Number of variables for which the gradient is computed (the first ones).
 * @param[in] idxVel - Variable index where the velocity starts (for the symmetry correction), -1 if none.
 * @param[in] which - 0 for the viscous gradient, 1 for the reconstruction gradient.
 * @param[out] firstCall - Set to true if the mirrors were created by this call (for messages).
 */
inline void ComputeGradientsLeastSquares(CFlowDeviceData& data, const CGeometryDevice& geometry, bool weighted,
                                         const su2double* hostPrimitives, size_t nPrimVar, su2double* hostGradient,
                                         size_t nVarStore, size_t nVarGrad, int idxVel, int which, bool& firstCall) {
  const size_t nPoint = geometry.GetnPoint(), nDim = geometry.GetnDim();

  auto& prim = data.primitives;
  auto& grad = data.gradient[which];
  firstCall = false;

  const bool firstOfAll = !prim.IsAttached() && !data.gradient[0].IsAttached() && !data.gradient[1].IsAttached();

  if (!prim.IsAttached() || prim.HostData() != hostPrimitives) {
    data.Attach(prim, const_cast<su2double*>(hostPrimitives), nPoint * nPrimVar, "primitives");
    firstCall = true;
  }
  if (!grad.IsAttached() || grad.HostData() != hostGradient) {
    data.Attach(grad, hostGradient, nPoint * nVarStore * nDim, which ? "reconstruction gradient" : "gradient");
    firstCall = true;
  }

  /*--- Warm-up on the still uninitialized buffers, see gg_gpu::ComputeGradientsGreenGauss. ---*/
  if (firstOfAll) {
    ComputeLeastSquaresDomain(geometry, nDim, weighted, prim.DeviceData(), nPrimVar, grad.DeviceData(), nVarStore, 0,
                              nVarGrad);
    gg_gpu::CorrectGradientsSymmetry(geometry, nDim, grad.DeviceWrite(), nVarStore, 0, nVarGrad, idxVel);
  }

  prim.ToDevice();

  ComputeLeastSquaresDomain(geometry, nDim, weighted, prim.DeviceRead(), nPrimVar, grad.DeviceWrite(), nVarStore, 0,
                            nVarGrad);
  gg_gpu::CorrectGradientsSymmetry(geometry, nDim, grad.DeviceWrite(), nVarStore, 0, nVarGrad, idxVel);

  auto& halo = geometry.GetHalo();
  const size_t up0 = halo.BytesUploaded(), down0 = halo.BytesDownloaded();
  const size_t nUp0 = halo.NumUploads(), nDown0 = halo.NumDownloads();
  halo.Exchange(grad.DeviceData(), nVarStore * nDim, nVarGrad * nDim);
  data.AddHaloTraffic(halo.BytesUploaded() - up0, halo.BytesDownloaded() - down0, halo.NumUploads() - nUp0,
                      halo.NumDownloads() - nDown0);

  grad.DeviceModified();
  if (nVarStore == nVarGrad) {
    grad.ToHost();
  } else {
    grad.ToHostRowPrefix(nVarGrad * nDim, nVarStore * nDim);
  }
}

}  // namespace ls_gpu

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
