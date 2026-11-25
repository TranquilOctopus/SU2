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
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../../Common/include/containers/CDeviceMirror.hpp"
#include "../../../Common/include/geometry/CGeometryDevice.hpp"
#include "../solvers/CFlowDeviceData.hpp"

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
 *                    and at least nPointDomain points (the halo rows are not touched).
 * @param[in] nVarStore - Number of variables per point stored in the gradient.
 * @param[in] varBegin - First variable for which the gradient is computed.
 * @param[in] varEnd - End (exclusive) of the range of variables.
 * @param[in] useMarker - One flag per marker, the markers with a zero are skipped.
 */
void ComputeGreenGaussDomain(const CGeometryDevice& geometry, size_t nDim, const su2double* field, size_t fieldLd,
                             su2double* grad, size_t nVarStore, size_t varBegin, size_t varEnd,
                             const std::vector<char>& useMarker);

/*!
 * @brief Correct the gradients on the symmetry plane and Euler wall markers on the device (the same as
 *        correctGradientsSymmetry does on the CPU: one launch per marker, in marker order, one thread per vertex, the
 *        per-vertex function is shared with the CPU).
 * @note Synchronous. Vertices on halo points are skipped (the halo rows are filled by the MPI communication that
 *       follows the correction, see CDeviceHaloExchange).
 * @param[in] geometry - Device copy of the geometry.
 * @param[in] nDim - Number of dimensions, 2 or 3.
 * @param[in,out] grad - Device array, row-major (point, variable, dimension), with nVarStore variables per point.
 * @param[in] nVarStore - Number of variables per point stored in the gradient.
 * @param[in] varBegin - First variable to correct.
 * @param[in] varEnd - End (exclusive) of the range of variables to correct.
 * @param[in] idxVel - Variable index where the velocity starts, -1 if all variables are scalars.
 */
void CorrectGradientsSymmetry(const CGeometryDevice& geometry, size_t nDim, su2double* grad, size_t nVarStore,
                              size_t varBegin, size_t varEnd, int idxVel);

/*!
 * @brief Run the first and second parts of the Green-Gauss gradient of the primitive variables on the GPU
 *        (volume and boundary contributions, then the symmetry and Euler wall corrections).
 * @details Launches the kernels, fills the halo rows of the gradient on the device with the values of the owner
 *          ranks (CDeviceHaloExchange of the geometry, the same messages as the CPU communication of the gradient: the
 *          first nVarGrad variables of every point) and downloads the final gradient of all the points. Nothing
 *          remains to be done on the CPU, the periodic communication is not needed since periodic markers are not
 *          supported by the offload. Must be called by a single thread of every rank (no OpenMP synchronization
 *          inside, MPI calls inside).
 *
 *          The primitives are NOT marked as modified here: the device copy is used as it is, and uploaded (once)
 *          only if the mirror says that the host copy is the newer one. Whoever writes the host primitives has to
 *          call data.PrimitivesModified() (the first call after the attachment uploads in any case).
 *
 *          When the gradient container stores more variables than are computed (nVarStore > nVarGrad), only the
 *          computed variables of every row are exchanged and downloaded, like the CPU communication does (it sends
 *          the nVarGrad first variables of the gradient, the others are not touched on either side): the other
 *          variables are never read by the kernels, and the host values are the current ones (they were
 *          downloaded, corrected, by an earlier call that computed them), while the device values may be stale, so
 *          downloading them would overwrite the corrections. Therefore nothing is uploaded either.
 * @param[in,out] data - Mirrors, attached on the first call (and again if the host arrays moved).
 * @param[in] geometry - Finest grid.
 * @param[in] markerSkipped - One flag per marker, true for the markers that do not contribute (internal, etc.).
 * @param[in] hostPrimitives - Host primitives, nPoint rows of length nPrimVar.
 * @param[in] nPrimVar - Row length of the primitives.
 * @param[in] hostGradient - Host gradient, nPoint rows of nVarStore x nDim.
 * @param[in] nVarStore - Number of variables per point stored in the gradient (its row length).
 * @param[in] nVarGrad - Number of variables for which the gradient is computed (the first ones, at most nVarStore),
 *                       the others are not modified.
 * @param[in] idxVel - Variable index where the velocity starts (for the symmetry correction), -1 if none.
 * @param[in] which - 0 for the viscous gradient, 1 for the reconstruction gradient.
 * @param[out] firstCall - Set to true if the mirrors were created by this call (for messages).
 */
inline void ComputeGradientsGreenGauss(CFlowDeviceData& data, const CGeometryDevice& geometry,
                                       const std::vector<char>& markerSkipped, const su2double* hostPrimitives,
                                       size_t nPrimVar, su2double* hostGradient, size_t nVarStore,
                                       size_t nVarGrad, int idxVel, int which, bool& firstCall) {
  const size_t nPoint = geometry.GetnPoint(), nDim = geometry.GetnDim();

  auto& prim = data.primitives;
  auto& grad = data.gradient[which];
  firstCall = false;

  const bool firstOfAll = !prim.IsAttached() && !data.gradient[0].IsAttached() && !data.gradient[1].IsAttached();

  if (!prim.IsAttached() || prim.HostData() != hostPrimitives) {
    data.Attach(prim, const_cast<su2double*>(hostPrimitives), nPoint * nPrimVar, "GG primitives");
    firstCall = true;
  }
  /*--- All the points: the halo rows are filled on the device by the halo exchange below. ---*/
  if (!grad.IsAttached() || grad.HostData() != hostGradient) {
    data.Attach(grad, hostGradient, nPoint * nVarStore * nDim,
                which ? "GG reconstruction gradient" : "GG gradient");
    firstCall = true;
  }

  data.useMarker.resize(markerSkipped.size());
  for (size_t i = 0; i < markerSkipped.size(); ++i) data.useMarker[i] = !markerSkipped[i];

  /*--- Warm-up: the first launch of the kernels is slow (module loading, first touch of the device buffers), so
   * it is done once here, on the still uninitialized device buffers (the results are garbage and are overwritten
   * by the real call below, the mirror states are not changed), instead of in the first timed call. ---*/
  if (firstOfAll) {
    ComputeGreenGaussDomain(geometry, nDim, prim.DeviceData(), nPrimVar, grad.DeviceData(), nVarStore, 0, nVarGrad,
                            data.useMarker);
    CorrectGradientsSymmetry(geometry, nDim, grad.DeviceWrite(), nVarStore, 0, nVarGrad, idxVel);
  }

  /*--- A no-op if the device copy is current (the usual case: the gradient of the same iteration was computed
   * before, or the solver uploaded it), see the description. ---*/
  prim.ToDevice();

  ComputeGreenGaussDomain(geometry, nDim, prim.DeviceRead(), nPrimVar, grad.DeviceWrite(), nVarStore, 0, nVarGrad,
                          data.useMarker);
  CorrectGradientsSymmetry(geometry, nDim, grad.DeviceWrite(), nVarStore, 0, nVarGrad, idxVel);

  /*--- The halo rows from the owners, the same values as the CPU communication gives. The pitch is the row of the
   * container, the width the variables that are computed (what the CPU sends for the gradient). ---*/
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

/*!
 * @brief Whether the check of the device halo exchange is requested (environment variable SU2_GPU_CHECK_COMMS set to
 *        something other than empty or 0, read once).
 * @details With the check on, the caller keeps a copy of the gradient that the device produced (halo rows received
 *          by the device exchange), then runs the CPU communication on the host gradient, and compares the halo rows
 *          with CheckHaloRows. It costs a copy of the gradient and a CPU exchange per call, so it is off by default.
 */
inline bool CheckCommsRequested() {
  static const bool requested = [] {
    const char* env = std::getenv("SU2_GPU_CHECK_COMMS");
    return env != nullptr && *env != '\0' && std::strcmp(env, "0") != 0;
  }();
  return requested;
}

/*!
 * @brief Compare, bit for bit, the halo rows that the device exchange produced with those of the CPU exchange. Ends
 *        the run with the details of the first differences if there is any.
 * @param[in] fromDevice - Copy of the host gradient as downloaded from the device.
 * @param[in] fromCpu - The host gradient after the CPU communication.
 * @param[in] nPointDomain - Number of domain points (the halo rows follow).
 * @param[in] nPoint - Number of points.
 * @param[in] pitch - Row length (nVarStore * nDim).
 * @param[in] width - Number of entries per row that the communication handles (nVarGrad * nDim).
 * @param[in] rank - MPI rank, for the messages.
 * @param[in] label - Name of the gradient, for the messages.
 */
inline void CheckHaloRows(const su2double* fromDevice, const su2double* fromCpu, size_t nPointDomain, size_t nPoint,
                          size_t pitch, size_t width, int rank, const std::string& label) {
  static size_t nCalls = 0, nValues = 0;
  size_t nBad = 0;
  for (size_t iPoint = nPointDomain; iPoint < nPoint; ++iPoint) {
    for (size_t j = 0; j < width; ++j) {
      const size_t k = iPoint * pitch + j;
      if (std::memcmp(&fromDevice[k], &fromCpu[k], sizeof(su2double)) != 0) {
        if (nBad < 5) {
          std::cerr << "SU2_GPU_CHECK_COMMS mismatch (rank " << rank << ", " << label << "): point " << iPoint
                    << " (halo row " << iPoint - nPointDomain << "), entry " << j << ": device " << std::scientific
                    << std::setprecision(17) << fromDevice[k] << ", CPU " << fromCpu[k] << std::endl;
        }
        ++nBad;
      }
    }
  }
  ++nCalls;
  nValues += (nPoint - nPointDomain) * width;
  if (nBad > 0) {
    SU2_MPI::Error("The halo rows of the gradient from the device exchange differ from the CPU exchange (" +
                       std::to_string(nBad) + " values, see above).",
                   CURRENT_FUNCTION);
  }
  if (nCalls <= 2 || nCalls % 200 == 0) {
    std::cout << "SU2_GPU_CHECK_COMMS rank " << rank << ": " << nCalls << " exchanges checked, " << nValues
              << " halo values bit-identical to the CPU exchange (last: " << label << ", row length " << pitch
              << ", exchanged " << width << ")." << std::endl;
  }
}

}  // namespace gg_gpu

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
