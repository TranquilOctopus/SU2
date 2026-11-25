/*!
 * \file computeGradientsGreenGauss.hpp
 * \brief Generic implementation of Green-Gauss gradient computation.
 * \note This allows the same implementation to be used for conservative
 *       and primitive variables of any solver.
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

#include <vector>
#include <algorithm>
#include <utility>

#include "../../../Common/include/parallelization/omp_structure.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"
#include "correctGradientsSymmetry.hpp"
#include "greenGaussPoint.hpp"

namespace detail {

/*!
 * \brief Preaccumulation policy for the host: forwards to the AD:: functions (empty in non-AD builds).
 * \note The policy is used by gg::greenGaussPoint at the places where the original code made the calls.
 */
struct AdPreacc {
  /*--- Cannot preaccumulate if hybrid parallel due to shared reading. ---*/
  static void Start() {
    if (omp_get_num_threads() == 1) AD::StartPreacc();
  }
  template <class T>
  static void In(T&& data) {
    AD::SetPreaccIn(std::forward<T>(data));
  }
  template <class T>
  static void InArray(const T& data, size_t size) {
    AD::SetPreaccIn(data, size);
  }
  template <class T>
  static void Out(T&& data) {
    AD::SetPreaccOut(std::forward<T>(data));
  }
  /*--- For the regions that were started unconditionally by the original code. ---*/
  static void StartUnguarded() { AD::StartPreacc(); }
  static void End() { AD::EndPreacc(); }
};

/*!
 * \brief Raw view of the geometry for greenGaussPoint, built from the host containers.
 */
inline auto makeGreenGaussGeomView(const CGeometry& geometry) {
  gg::GeomView<su2double, CCompressedSparsePatternUL::IndexType, CCompressedSparsePatternL::IndexType> view;
  const auto* nodes = geometry.nodes;
  const auto& points = nodes->GetPoints();
  if (points.empty() || geometry.edges == nullptr) return view;

  view.nbrOuter = points.outerPtr();
  view.nbrIdx = points.innerIdx();
  view.edgeIdx = nodes->GetEdges().innerIdx();
  const auto& normal = geometry.edges->GetNormal();
  view.normal = normal.data();
  view.normalLd = normal.cols();
  view.volume = &nodes->GetVolume(0);
  view.periodicVolume = &nodes->GetPeriodicVolume(0);
  return view;
}

/*!
 * \brief First part of the Green-Gauss gradient: the volume (edge) and boundary marker contributions at all
 *        domain points. Halo points, symmetry corrections, periodic and MPI communications are not handled,
 *        see computeGradientsGreenGaussFinish. This is the part that can run on the GPU.
 * \ingroup FvmAlgos
 * \note Template nDim to allow efficient unrolling of inner loops.
 * \note Gradients can be computed only for a contiguous range of variables, defined
 *       by [varBegin, varEnd[ (e.g. 0,1 computes the gradient of the 1st variable).
 * \param[in] geometry - Geometric grid properties.
 * \param[in] config - Configuration of the problem, used to identify types of boundaries.
 * \param[in] field - Generic object implementing operator (iPoint, iVar).
 * \param[in] varBegin - Index of first variable for which to compute the gradient.
 * \param[in] varEnd - Index of last variable for which to compute the gradient.
 * \param[out] gradient - Generic object implementing operator (iPoint, iVar, iDim).
 */
template <size_t nDim, class FieldType, class GradientType>
void computeGradientsGreenGaussDomain(CGeometry& geometry, const CConfig& config, const FieldType& field,
                                      const size_t varBegin, const size_t varEnd, GradientType& gradient) {
  const size_t nPointDomain = geometry.GetnPointDomain();

#ifdef HAVE_OMP
  constexpr size_t OMP_MAX_CHUNK = 512;

  const auto chunkSize = computeStaticChunkSize(nPointDomain, omp_get_max_threads(), OMP_MAX_CHUNK);
#endif

  static constexpr size_t MAXNVAR = 20;

  const auto geomView = makeGreenGaussGeomView(geometry);

  /*--- For each (non-halo) volume integrate over its faces (edges). ---*/

  SU2_OMP_FOR_DYN(chunkSize)
  for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint) {
    gg::greenGaussPoint<nDim, su2double, AdPreacc>(iPoint, geomView, field, varBegin, varEnd, gradient);
  }
  END_SU2_OMP_FOR

  su2double flux[MAXNVAR] = {0.0};

  /*--- Add edges of markers that contribute to the gradients ---*/
  for (size_t iMarker = 0; iMarker < geometry.GetnMarker(); ++iMarker) {
    if ((config.GetMarker_All_KindBC(iMarker) != INTERNAL_BOUNDARY) &&
        (config.GetMarker_All_KindBC(iMarker) != NEARFIELD_BOUNDARY) &&
        (config.GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) {
      /*--- Work is shared in inner loop as two markers
       *    may try to update the same point. ---*/

      SU2_OMP_FOR_STAT(32)
      for (size_t iVertex = 0; iVertex < geometry.GetnVertex(iMarker); ++iVertex) {
        size_t iPoint = geometry.vertex[iMarker][iVertex]->GetNode();
        auto nodes = geometry.nodes;

        /*--- Halo points do not need to be considered. ---*/

        if (!nodes->GetDomain(iPoint)) continue;

        su2double volume = nodes->GetVolume(iPoint) + nodes->GetPeriodicVolume(iPoint);
        const auto area = geometry.vertex[iMarker][iVertex]->GetNormal();

        for (size_t iVar = varBegin; iVar < varEnd; iVar++)
          flux[iVar] = field(iPoint,iVar) / volume;

        for (size_t iVar = varBegin; iVar < varEnd; iVar++) {
          for (size_t iDim = 0; iDim < nDim; iDim++) {
            gradient(iPoint, iVar, iDim) -= flux[iVar] * area[iDim];
          }
        } // loop over variables
      } // vertices
      END_SU2_OMP_FOR
    } //found right marker
  } // iMarkers
}

/*!
 * \brief Last part of the Green-Gauss gradient: periodic contributions and the MPI communications (always on
 *        the CPU). Does nothing if no solver is given.
 * \ingroup FvmAlgos
 * \param[in] solver - Optional, solver associated with the field (used only for MPI).
 * \param[in] kindMpiComm - Type of MPI communication required.
 * \param[in] kindPeriodicComm - Type of periodic communication required.
 * \param[in] geometry - Geometric grid properties.
 * \param[in] config - Configuration of the problem, used to identify types of boundaries.
 */
inline void computeGradientsGreenGaussComms(CSolver* solver, MPI_QUANTITIES kindMpiComm,
                                            PERIODIC_QUANTITIES kindPeriodicComm, CGeometry& geometry,
                                            const CConfig& config) {
  /*--- If no solver was provided we do not communicate ---*/

  if (solver == nullptr) return;

  /*--- Account for periodic contributions. ---*/

  for (size_t iPeriodic = 1; iPeriodic <= config.GetnMarker_Periodic() / 2; ++iPeriodic) {
    solver->InitiatePeriodicComms(&geometry, &config, iPeriodic, kindPeriodicComm);
    solver->CompletePeriodicComms(&geometry, &config, iPeriodic, kindPeriodicComm);
  }

  /*--- Obtain the gradients at halo points from the MPI ranks that own them. ---*/

  solver->InitiateComms(&geometry, &config, kindMpiComm);
  solver->CompleteComms(&geometry, &config, kindMpiComm);
}

/*!
 * \brief Second part of the Green-Gauss gradient: corrections for symmetry planes and Euler walls, then the
 *        periodic contributions and the MPI communications (see computeGradientsGreenGaussComms).
 * \ingroup FvmAlgos
 * \param[in] solver - Optional, solver associated with the field (used only for MPI).
 * \param[in] kindMpiComm - Type of MPI communication required.
 * \param[in] kindPeriodicComm - Type of periodic communication required.
 * \param[in] geometry - Geometric grid properties.
 * \param[in] config - Configuration of the problem, used to identify types of boundaries.
 * \param[in] varBegin - Index of first variable for which the gradient was computed.
 * \param[in] varEnd - Index of last variable for which the gradient was computed.
 * \param[in] idxVel - Index of velocity, or -1 if no velocity present.
 * \param[in,out] gradient - Generic object implementing operator (iPoint, iVar, iDim).
 */
template <size_t nDim, class GradientType>
void computeGradientsGreenGaussFinish(CSolver* solver, MPI_QUANTITIES kindMpiComm,
                                      PERIODIC_QUANTITIES kindPeriodicComm, CGeometry& geometry,
                                      const CConfig& config, const size_t varBegin, const size_t varEnd,
                                      const int idxVel, GradientType& gradient) {
  /*--- Compute the corrections for symmetry planes and Euler walls. ---*/

  correctGradientsSymmetry<nDim>(geometry, config, varBegin, varEnd, idxVel, gradient);

  computeGradientsGreenGaussComms(solver, kindMpiComm, kindPeriodicComm, geometry, config);
}

/*!
 * \brief Compute the gradient of a field using the Green-Gauss theorem (both parts, on the CPU).
 * \ingroup FvmAlgos
 * \note See computeGradientsGreenGaussDomain and computeGradientsGreenGaussFinish for the two parts.
 * \note The function uses an optional solver object to perform communications, if
 *       none (nullptr) is provided the function does not fail (the objective of
 *       this is to improve test-ability).
 * \param[in] solver - Optional, solver associated with the field (used only for MPI).
 * \param[in] kindMpiComm - Type of MPI communication required.
 * \param[in] kindPeriodicComm - Type of periodic communication required.
 * \param[in] geometry - Geometric grid properties.
 * \param[in] config - Configuration of the problem, used to identify types of boundaries.
 * \param[in] field - Generic object implementing operator (iPoint, iVar).
 * \param[in] varBegin - Index of first variable for which to compute the gradient.
 * \param[in] varEnd - Index of last variable for which to compute the gradient.
 * \param[in] idxVel - Index of velocity, or -1 if no velocity present.
 * \param[out] gradient - Generic object implementing operator (iPoint, iVar, iDim).
 */
template <size_t nDim, class FieldType, class GradientType>
void computeGradientsGreenGauss(CSolver* solver, MPI_QUANTITIES kindMpiComm, PERIODIC_QUANTITIES kindPeriodicComm,
                                CGeometry& geometry, const CConfig& config, const FieldType& field,
                                const size_t varBegin, const size_t varEnd, const int idxVel, GradientType& gradient) {
  computeGradientsGreenGaussDomain<nDim>(geometry, config, field, varBegin, varEnd, gradient);
  computeGradientsGreenGaussFinish<nDim>(solver, kindMpiComm, kindPeriodicComm, geometry, config, varBegin, varEnd,
                                         idxVel, gradient);
}
}  // namespace detail



/*!
 * \brief Instantiations for 2D and 3D.
 * \ingroup FvmAlgos
 */
template <class FieldType, class GradientType>
void computeGradientsGreenGauss(CSolver* solver, MPI_QUANTITIES kindMpiComm, PERIODIC_QUANTITIES kindPeriodicComm,
                                CGeometry& geometry, const CConfig& config, const FieldType& field,
                                const size_t varBegin, const size_t varEnd, const int idxVel, GradientType& gradient) {
  switch (geometry.GetnDim()) {
    case 2:
      detail::computeGradientsGreenGauss<2>(solver, kindMpiComm, kindPeriodicComm, geometry, config, field, varBegin,
                                            varEnd, idxVel, gradient);
      break;
    case 3:
      detail::computeGradientsGreenGauss<3>(solver, kindMpiComm, kindPeriodicComm, geometry, config, field, varBegin,
                                            varEnd, idxVel, gradient);
      break;
    default:
      SU2_MPI::Error("Too many dimensions to compute gradients.", CURRENT_FUNCTION);
      break;
  }
}

/*!
 * \brief Second part of the Green-Gauss gradient for a runtime number of dimensions, see
 *        detail::computeGradientsGreenGaussFinish. For use after the first part was computed elsewhere (the GPU).
 */
template <class GradientType>
void computeGradientsGreenGaussFinish(CSolver* solver, MPI_QUANTITIES kindMpiComm,
                                      PERIODIC_QUANTITIES kindPeriodicComm, CGeometry& geometry,
                                      const CConfig& config, const size_t varBegin, const size_t varEnd,
                                      const int idxVel, GradientType& gradient) {
  switch (geometry.GetnDim()) {
    case 2:
      detail::computeGradientsGreenGaussFinish<2>(solver, kindMpiComm, kindPeriodicComm, geometry, config, varBegin,
                                                  varEnd, idxVel, gradient);
      break;
    case 3:
      detail::computeGradientsGreenGaussFinish<3>(solver, kindMpiComm, kindPeriodicComm, geometry, config, varBegin,
                                                  varEnd, idxVel, gradient);
      break;
    default:
      SU2_MPI::Error("Too many dimensions to compute gradients.", CURRENT_FUNCTION);
      break;
  }
}
