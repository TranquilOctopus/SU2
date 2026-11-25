/*!
 * @file computeGradientsGreenGaussGPU.hpp
 * @brief GPU version of the first part of the Green-Gauss gradient (volume and boundary marker contributions),
 *        see computeGradientsGreenGauss.hpp for the CPU version and the second part (always on the CPU).
 * @note The kernels (computeGradientsGreenGaussGPU.cu) use the same per-point function as the CPU
 *       (greenGaussPoint.hpp) and are compiled without FMA contraction, see the comments there.
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

namespace gg_gpu {

/*!
 * @brief Compute the volume and boundary marker contributions of the Green-Gauss gradient at all domain points
 *        on the device (the same as detail::computeGradientsGreenGaussDomain does on the CPU).
 * @note Synchronous: the results are complete when the function returns. Errors end the run (gpuErrChk).
 *       Halo rows of the gradient are not touched.
 * @param[in] geometry - Device copy of the geometry.
 * @param[in] nDim - Number of dimensions, 2 or 3.
 * @param[in] field - Device array, row-major, one row of length fieldLd per point (all points, incl. halos).
 * @param[in] fieldLd - Row length of the field.
 * @param[out] grad - Device array, row-major (point, variable, dimension), with nVarStore variables per point
 *                    and at least nPointDomain points.
 * @param[in] nVarStore - Number of variables per point stored in the gradient.
 * @param[in] varBegin - First variable for which the gradient is computed.
 * @param[in] varEnd - End (exclusive) of the range of variables.
 * @param[in] useMarker - One flag per marker, the markers with a zero are skipped.
 */
void ComputeGreenGaussDomain(const CGeometryDevice& geometry, size_t nDim, const su2double* field, size_t fieldLd,
                             su2double* grad, size_t nVarStore, size_t varBegin, size_t varEnd,
                             const std::vector<char>& useMarker);

/*!
 * @brief Device mirrors of the primitive variables and of the gradients (viscous and, if different,
 *        reconstruction) of one flow solver. Created lazily, owned by the solver.
 */
struct GreenGaussDeviceData {
  CDeviceMirror<su2double> primitives;
  CDeviceMirror<su2double> gradient[2]; /*!< @brief [0] viscous gradient, [1] reconstruction gradient. */
  std::vector<char> useMarker;
};

/*!
 * @brief Run the first part of the Green-Gauss gradient of the primitive variables on the GPU.
 * @details Uploads the primitives (all points), launches the kernels and downloads the gradient of the domain
 *          points, which is a contiguous prefix of the host array, so the halo rows are left alone. Must be called
 *          by a single thread (no OpenMP synchronization inside). The caller then runs the CPU part (second part).
 * @param[in,out] data - Mirrors, attached on the first call (and again if the host arrays moved).
 * @param[in] geometry - Finest grid.
 * @param[in] markerSkipped - One flag per marker, true for the markers that do not contribute (internal, etc.).
 * @param[in] hostPrimitives - Host primitives, nPoint rows of length nPrimVar.
 * @param[in] nPrimVar - Row length of the primitives.
 * @param[in] hostGradient - Host gradient, nPoint rows of nVarStore x nDim.
 * @param[in] nVarStore - Number of variables per point stored in the gradient (its row length).
 * @param[in] nVarGrad - Number of variables for which the gradient is computed (the first ones, at most nVarStore),
 *                       the others are not modified.
 * @param[in] which - 0 for the viscous gradient, 1 for the reconstruction gradient.
 * @param[in] firstCall - Set to true if the mirrors were created by this call (for messages).
 */
inline void ComputeGradientsGreenGauss(GreenGaussDeviceData& data, const CGeometryDevice& geometry,
                                       const std::vector<char>& markerSkipped, const su2double* hostPrimitives,
                                       size_t nPrimVar, su2double* hostGradient, size_t nVarStore,
                                       size_t nVarGrad, int which, bool& firstCall) {
  const size_t nPoint = geometry.GetnPoint(), nPointDomain = geometry.GetnPointDomain(), nDim = geometry.GetnDim();

  auto& prim = data.primitives;
  auto& grad = data.gradient[which];
  firstCall = false;

  if (!prim.IsAttached() || prim.HostData() != hostPrimitives) {
    prim.Attach(const_cast<su2double*>(hostPrimitives), nPoint * nPrimVar, "GG primitives");
    firstCall = true;
  }
  /*--- Only the domain points are attached: they come first in the storage, and the halo rows must keep the
   * values that are on the host (they are only overwritten by the MPI communication after this). ---*/
  if (!grad.IsAttached() || grad.HostData() != hostGradient) {
    grad.Attach(hostGradient, nPointDomain * nVarStore * nDim, which ? "GG reconstruction gradient" : "GG gradient");
    firstCall = true;
  }

  data.useMarker.resize(markerSkipped.size());
  for (size_t i = 0; i < markerSkipped.size(); ++i) data.useMarker[i] = !markerSkipped[i];

  prim.HostModified();
  prim.ToDevice();

  /*--- Variables that are stored but not computed are copied back too, so they must be current on the device. ---*/
  if (nVarStore != nVarGrad) {
    grad.HostModified();
    grad.ToDevice();
  }

  ComputeGreenGaussDomain(geometry, nDim, prim.DeviceRead(), nPrimVar, grad.DeviceWrite(), nVarStore, 0, nVarGrad,
                          data.useMarker);

  grad.DeviceModified();
  grad.ToHost();
}

}  // namespace gg_gpu

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
