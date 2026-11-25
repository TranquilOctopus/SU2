/*!
 * @file computeGradientsGreenGaussGPU.cu
 * @brief CUDA kernels of the Green-Gauss gradient (volume and boundary marker contributions).
 * @note The results are bit-identical to those of the CPU implementation (checked by the unit test
 *       UnitTests/SU2_CFD/gradients_gpu.cpp), which depends on how the host compiler rounds that code. The host
 *       build (-O3 -march=native -ffast-math with GCC) (1) contracts "grad +=/-= flux * area" into fused
 *       multiply-adds, so this file is compiled with FMA contraction on (-fmad=true, see SU2_CFD/src/meson.build,
 *       -fmad=false would NOT match the host), and (2) turns the division by the control volume in the marker
 *       loop into a multiplication by a reciprocal computed once (-freciprocal-math), which the boundary kernel
 *       below does explicitly. If the host flags or compiler change, rerun the unit test.
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

#include "../../include/gradients/computeGradientsGreenGaussGPU.hpp"

#ifdef SU2_ENABLE_GEOMETRY_DEVICE

#include <cstdint>

#include "../../../Common/include/parallelization/mpi_structure.hpp"

#include "../../include/gradients/greenGaussPoint.hpp"

namespace gg_gpu {
namespace {

constexpr int BLOCK_SIZE = 128;

/*--- One thread per domain point, the body is the function shared with the CPU. ---*/
template <size_t nDim>
__global__ void VolumeKernel(CGeometryDevice::View geom, const su2double* field, size_t fieldLd, su2double* grad,
                             size_t nVarGrad, size_t varBegin, size_t varEnd, size_t nPointDomain) {
  const size_t iPoint = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (iPoint >= nPointDomain) return;

  const gg::FieldView<su2double> fieldView{field, fieldLd};
  gg::GradView<su2double, nDim> gradView{grad, nVarGrad};
  gg::greenGaussPoint<nDim, su2double, gg::NoPreacc>(iPoint, geom, fieldView, varBegin, varEnd, gradView);
}

/*--- One thread per vertex of one marker, the same operations as the marker loop of the CPU implementation
 *    (computeGradientsGreenGaussDomain). A point appears at most once per marker, so there are no races, the
 *    markers are processed one after the other (launches on the same stream) like on the CPU. ---*/
template <size_t nDim>
__global__ void BoundaryKernel(CGeometryDevice::MarkerView marker, const uint8_t* domain, const su2double* volumes,
                               const su2double* periodicVolumes, const su2double* field, size_t fieldLd,
                               su2double* grad, size_t nVarGrad, size_t varBegin, size_t varEnd) {
  const size_t iVertex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (iVertex >= marker.nVertex) return;

  const size_t iPoint = marker.node[iVertex];

  /*--- Halo points do not need to be considered. ---*/
  if (!domain[iPoint]) return;

  const su2double volume = volumes[iPoint] + periodicVolumes[iPoint];
  const su2double* area = marker.normal + iVertex * nDim;

  const gg::FieldView<su2double> fieldView{field, fieldLd};
  gg::GradView<su2double, nDim> gradView{grad, nVarGrad};

  /*--- The host compiler evaluates "field / volume" as field * (1 / volume), see the note at the top. ---*/
  const su2double rVolume = 1.0 / volume;

  for (size_t iVar = varBegin; iVar < varEnd; iVar++) {
    const su2double flux = fieldView(iPoint, iVar) * rVolume;
    for (size_t iDim = 0; iDim < nDim; iDim++) {
      gradView(iPoint, iVar, iDim) -= flux * area[iDim];
    }
  }
}

template <size_t nDim>
void Launch(const CGeometryDevice& geometry, const su2double* field, size_t fieldLd, su2double* grad,
            size_t nVarGrad, size_t varBegin, size_t varEnd, const std::vector<char>& useMarker) {
  const size_t nPointDomain = geometry.GetnPointDomain();
  const auto geom = geometry.GetGeomView();

  if (nPointDomain > 0) {
    const unsigned blocks = static_cast<unsigned>((nPointDomain + BLOCK_SIZE - 1) / BLOCK_SIZE);
    VolumeKernel<nDim><<<blocks, BLOCK_SIZE>>>(geom, field, fieldLd, grad, nVarGrad, varBegin, varEnd, nPointDomain);
    gpuErrChk(cudaGetLastError());
  }

  const uint8_t* domain = geometry.GetDomain();

  for (size_t iMarker = 0; iMarker < geometry.GetnMarker(); ++iMarker) {
    if (iMarker < useMarker.size() && !useMarker[iMarker]) continue;
    const auto marker = geometry.GetMarker(iMarker);
    if (marker.nVertex == 0) continue;
    const unsigned blocks = static_cast<unsigned>((marker.nVertex + BLOCK_SIZE - 1) / BLOCK_SIZE);
    BoundaryKernel<nDim><<<blocks, BLOCK_SIZE>>>(marker, domain, geom.volume, geom.periodicVolume, field, fieldLd,
                                                 grad, nVarGrad, varBegin, varEnd);
    gpuErrChk(cudaGetLastError());
  }

  gpuErrChk(cudaDeviceSynchronize());
}

}  // namespace

void ComputeGreenGaussDomain(const CGeometryDevice& geometry, size_t nDim, const su2double* field, size_t fieldLd,
                             su2double* grad, size_t nVarGrad, size_t varBegin, size_t varEnd,
                             const std::vector<char>& useMarker) {
  if (nDim == 2) {
    Launch<2>(geometry, field, fieldLd, grad, nVarGrad, varBegin, varEnd, useMarker);
  } else if (nDim == 3) {
    Launch<3>(geometry, field, fieldLd, grad, nVarGrad, varBegin, varEnd, useMarker);
  } else {
    SU2_MPI::Error("The Green-Gauss gradient on the GPU supports only 2 or 3 dimensions.", CURRENT_FUNCTION);
  }
}

}  // namespace gg_gpu

#endif  // SU2_ENABLE_GEOMETRY_DEVICE
