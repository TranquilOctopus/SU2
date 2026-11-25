/*!
 * \file computeGradientsLeastSquares.hpp
 * \brief Generic implementation of Least-Squares gradient computation.
 * \note This allows the same implementation to be used for conservative
 *       and primitive variables of any solver.
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

#include "../../../Common/include/parallelization/omp_structure.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"
#include "computeGradientsGreenGauss.hpp"
#include "correctGradientsSymmetry.hpp"
#include "leastSquaresPoint.hpp"

namespace detail {

/*!
 * \brief Raw view of the geometry for ls::leastSquaresPoint (neighbors and coordinates), built from the host containers.
 */
inline auto makeLeastSquaresGeomView(const CGeometry& geometry) {
  gg::GeomView<su2double, CCompressedSparsePatternUL::IndexType, CCompressedSparsePatternL::IndexType> view;
  const auto* nodes = geometry.nodes;
  const auto& points = nodes->GetPoints();
  view.nbrOuter = points.outerPtr();
  view.nbrIdx = points.innerIdx();
  const auto& coord = nodes->GetCoord();
  view.coord = coord.data();
  view.coordLd = coord.cols();
  return view;
}

/*!
 * \brief Compute the gradient of a field using inverse-distance-weighted or
 *        unweighted Least-Squares approximation.
 * \ingroup FvmAlgos
 * \note See notes from computeGradientsGreenGauss.hpp.
 * \param[in] solver - Optional, solver associated with the field (used only for MPI).
 * \param[in] kindMpiComm - Type of MPI communication required.
 * \param[in] kindPeriodicComm - Type of periodic communication required.
 * \param[in] geometry - Geometric grid properties.
 * \param[in] weighted - Use inverse-distance weights.
 * \param[in] config - Configuration of the problem, used to identify types of boundaries.
 * \param[in] field - Generic object implementing operator (iPoint, iVar).
 * \param[in] varBegin - Index of first variable for which to compute the gradient.
 * \param[in] varEnd - Index of last variable for which to compute the gradient.
 * \param[in] idxVel - Index to velocity, -1 if no velocity is present in the solver.
 * \param[out] gradient - Generic object implementing operator (iPoint, iVar, iDim).
 * \param[out] Rmatrix - Generic object implementing operator (iPoint, iDim, iDim).
 */
template<size_t nDim, class FieldType, class GradientType, class RMatrixType>
void computeGradientsLeastSquares(CSolver* solver,
                                  MPI_QUANTITIES kindMpiComm,
                                  PERIODIC_QUANTITIES kindPeriodicComm,
                                  CGeometry& geometry,
                                  const CConfig& config,
                                  bool weighted,
                                  const FieldType& field,
                                  const size_t varBegin,
                                  const size_t varEnd,
                                  const int idxVel,
                                  GradientType& gradient,
                                  RMatrixType& Rmatrix)
{
  const bool periodic = (solver != nullptr) && (config.GetnMarker_Periodic() > 0);

  const size_t nPointDomain = geometry.GetnPointDomain();

#ifdef HAVE_OMP
  constexpr size_t OMP_MAX_CHUNK = 512;

  size_t chunkSize = computeStaticChunkSize(nPointDomain,
                     omp_get_max_threads(), OMP_MAX_CHUNK);
#endif

  const auto geomView = makeLeastSquaresGeomView(geometry);

  /*--- First loop over non-halo points of the grid. ---*/

  SU2_OMP_FOR_DYN(chunkSize)
  for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint)
  {
    if (periodic)
    {
      /*--- A second loop is required after periodic comms, checkpoint the preacc. ---*/

      ls::leastSquaresAccumulate<nDim, AdPreacc>(iPoint, geomView, weighted, field, varBegin, varEnd, gradient,
                                                 Rmatrix);

      for (size_t iDim = 0; iDim < nDim; ++iDim)
        for (size_t jDim = 0; jDim < nDim; ++jDim)
          AD::SetPreaccOut(Rmatrix(iPoint, iDim, jDim));

      for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
        for (size_t iDim = 0; iDim < nDim; ++iDim)
          AD::SetPreaccOut(gradient(iPoint, iVar, iDim));

      AD::EndPreacc();
    }
    else {
      /*--- Periodic comms are not needed, solve the LS problem for iPoint. ---*/

      ls::leastSquaresPoint<nDim, AdPreacc>(iPoint, geomView, weighted, field, varBegin, varEnd, gradient, Rmatrix);
    }
  }
  END_SU2_OMP_FOR

  /*--- Correct the gradient values across any periodic boundaries. ---*/

  if (periodic)
  {
    for (size_t iPeriodic = 1; iPeriodic <= config.GetnMarker_Periodic()/2; ++iPeriodic)
    {
      solver->InitiatePeriodicComms(&geometry, &config, iPeriodic, kindPeriodicComm);
      solver->CompletePeriodicComms(&geometry, &config, iPeriodic, kindPeriodicComm);
    }

    /*--- Second loop over points of the grid to compute final gradient. ---*/

    SU2_OMP_FOR_DYN(chunkSize)
    for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint)
      ls::solveLeastSquares<nDim, true, AdPreacc>(iPoint, varBegin, varEnd, Rmatrix, gradient);
    END_SU2_OMP_FOR
  }

  /* --- compute the corrections for symmetry planes and Euler walls. --- */

  correctGradientsSymmetry<nDim>(geometry, config, varBegin, varEnd, idxVel, gradient);

  /*--- If no solver was provided we do not communicate ---*/

  if (solver != nullptr)
  {
    /*--- Obtain the gradients at halo points from the MPI ranks that own them. ---*/

    solver->InitiateComms(&geometry, &config, kindMpiComm);
    solver->CompleteComms(&geometry, &config, kindMpiComm);
  }

}
} // end namespace

/*!
 * \brief Instantiations for 2D and 3D.
 * \ingroup FvmAlgos
 */
template<class FieldType, class GradientType, class RMatrixType>
void computeGradientsLeastSquares(CSolver* solver,
                                  MPI_QUANTITIES kindMpiComm,
                                  PERIODIC_QUANTITIES kindPeriodicComm,
                                  CGeometry& geometry,
                                  const CConfig& config,
                                  bool weighted,
                                  const FieldType& field,
                                  const size_t varBegin,
                                  const size_t varEnd,
                                  const int idxVel,
                                  GradientType& gradient,
                                  RMatrixType& Rmatrix) {
  switch (geometry.GetnDim()) {
  case 2:
    detail::computeGradientsLeastSquares<2>(solver, kindMpiComm, kindPeriodicComm, geometry, config,
                                            weighted, field, varBegin, varEnd, idxVel, gradient, Rmatrix);
    break;
  case 3:
    detail::computeGradientsLeastSquares<3>(solver, kindMpiComm, kindPeriodicComm, geometry, config,
                                            weighted, field, varBegin, varEnd, idxVel, gradient, Rmatrix);
    break;
  default:
    SU2_MPI::Error("Too many dimensions to compute gradients.", CURRENT_FUNCTION);
    break;
  }
}
