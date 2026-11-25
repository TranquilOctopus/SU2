/*!
 * \file limiterPoint.hpp
 * \brief The slope limiter of one point, shared by the CPU implementation (computeLimiters_impl) and the CUDA kernel.
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

#include <cstddef>

#include "../../../Common/include/code_config.hpp"
#include "../../../Common/include/geometry/GeomView.hpp"
#include "../gradients/greenGaussPoint.hpp"
#include "CLimiterDetails.hpp"

namespace lim {

constexpr size_t MAXNVAR = LIMITER_MAX_NVAR;

/*!
 * \brief Compute the min/max of the field over the neighbors of one point, the min/max of the reconstruction
 *        increments over the edges, and the limiter of every variable of the point (the body of the loop over the
 *        points of computeLimiters_impl).
 * \ingroup FvmAlgos
 * \note The expressions and the order of the neighbors (storage order of the compressed sparse pattern) are those of
 *       the original CPU implementation. The calls to the preaccumulation policy are at the places where the
 *       original code called AD::StartPreacc/SetPreaccIn/SetPreaccOut/EndPreacc.
 * \tparam Preacc - Preaccumulation policy (see gg::NoPreacc and, for the host, detail::AdPreacc).
 * \param[in] iPoint - Point (must be a domain point).
 * \param[in] g - Geometry view with the neighbors (nbrOuter, nbrIdx) and the coordinates (coord, coordLd).
 * \param[in] limiterDetails - Object of the CLimiterDetails specialization of the limiter, after "preprocess".
 * \param[in,out] geoContext - Passed to the geometricFactor of the details (the geometry on the CPU).
 * \param[in] periodic - If true fieldMin/fieldMax hold the min/max over the periodic matches on entry, otherwise
 *            they are initialized with the value of the field at the point.
 * \param[in] periodicProj - Min then max (starting at periodicHalf) of the projections over the edges of the
 *            periodic matches of the point, nullptr if there are none.
 * \param[in] periodicHalf - Offset of the maxima in periodicProj.
 * \param[in] varBegin - First variable.
 * \param[in] varEnd - End of the variable range (exclusive), at most MAXNVAR.
 * \param[in] umusclKappa - Blending parameter for U-MUSCL reconstruction.
 * \param[in] field - Object implementing operator (iPoint, iVar).
 * \param[in] gradient - Object implementing operator [iPoint][iVar] which returns a pointer to the nDim components.
 * \param[out] fieldMin - Object implementing operator (iPoint, iVar), minimum of the field over the neighbors.
 * \param[out] fieldMax - As above, maximum.
 * \param[out] limiter - Object implementing operator (iPoint, iVar), the limiter of the point.
 */
template <size_t nDim, class Preacc, class Details, class GeoContext, class GeomView, class FieldType,
          class GradientType, class MinMaxType, class LimiterType>
SU2_HOST_DEVICE FORCEINLINE void limiterPoint(size_t iPoint, const GeomView& g, const Details& limiterDetails,
                                              GeoContext& geoContext, bool periodic, const su2double* periodicProj,
                                              size_t periodicHalf, size_t varBegin, size_t varEnd,
                                              su2double umusclKappa, const FieldType& field,
                                              const GradientType& gradient, MinMaxType& fieldMin,
                                              MinMaxType& fieldMax, LimiterType& limiter) {
  const su2double* coord_i = g.coord + iPoint * g.coordLd;

  /*--- Cannot preaccumulate if hybrid parallel due to shared reading. ---*/
  Preacc::Start();
  Preacc::InArray(coord_i, nDim);

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
    Preacc::In(field(iPoint, iVar));

    if (periodic) {
      /*--- Started outside loop, so counts as input. ---*/
      Preacc::In(fieldMax(iPoint, iVar));
      Preacc::In(fieldMin(iPoint, iVar));
    } else {
      /*--- Initialize min/max now for iPoint if not periodic. ---*/
      fieldMax(iPoint, iVar) = field(iPoint, iVar);
      fieldMin(iPoint, iVar) = field(iPoint, iVar);
    }

    for (size_t iDim = 0; iDim < nDim; ++iDim) Preacc::In(gradient[iPoint][iVar][iDim]);
  }

  /*--- Initialize min/max projection out of iPoint. ---*/

  su2double projMax[MAXNVAR], projMin[MAXNVAR];

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar) projMax[iVar] = projMin[iVar] = 0.0;

  if (periodicProj != nullptr) {
    /*--- Start from the min/max over the edges of the periodic matches. ---*/

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
      const su2double& periodicMin = periodicProj[iVar];
      const su2double& periodicMax = periodicProj[periodicHalf + iVar];
      Preacc::In(periodicMin);
      Preacc::In(periodicMax);
      projMin[iVar] = periodicMin;
      projMax[iVar] = periodicMax;
    }
  }

  /*--- Compute max/min projection and values over direct neighbors. ---*/

  const size_t rowBegin = g.nbrOuter[iPoint];
  const size_t nNeigh = g.nbrOuter[iPoint + 1] - rowBegin;

  for (size_t iNeigh = 0; iNeigh < nNeigh; ++iNeigh) {
    const size_t jPoint = g.nbrIdx[rowBegin + iNeigh];

    const su2double* coord_j = g.coord + jPoint * g.coordLd;
    Preacc::InArray(coord_j, nDim);

    /*--- Distance vector from iPoint to face (middle of the edge). ---*/

    su2double dist_ij[nDim] = {0.0};
    for (size_t iDim = 0; iDim < nDim; ++iDim) dist_ij[iDim] = 0.5 * (coord_j[iDim] - coord_i[iDim]);

    /*--- Project each variable, update min/max. ---*/

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
      Preacc::In(field(jPoint, iVar));
      const su2double proj = LimiterHelpers<>::reconstructionIncrement(
          nDim, dist_ij, gradient[iPoint][iVar], field(iPoint, iVar), field(jPoint, iVar), umusclKappa);

      projMax[iVar] = max(projMax[iVar], proj);
      projMin[iVar] = min(projMin[iVar], proj);

      fieldMax(iPoint, iVar) = max(fieldMax(iPoint, iVar), field(jPoint, iVar));
      fieldMin(iPoint, iVar) = min(fieldMin(iPoint, iVar), field(jPoint, iVar));
    }
  }

  /*--- Compute the geometric factor. ---*/

  su2double geoFactor = limiterDetails.geometricFactor(iPoint, geoContext);

  /*--- Final limiter computation for each variable, get the min limiter
   *    out of the positive/negative projections and deltas. ---*/

  for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
    su2double limMax = limiterDetails.template limiterFunction<Preacc>(iVar, projMax[iVar],
                                                                       fieldMax(iPoint, iVar) - field(iPoint, iVar));

    su2double limMin = limiterDetails.template limiterFunction<Preacc>(iVar, projMin[iVar],
                                                                       fieldMin(iPoint, iVar) - field(iPoint, iVar));

    limiter(iPoint, iVar) = geoFactor * min(limMax, limMin);

    Preacc::Out(limiter(iPoint, iVar));
  }

  Preacc::End();
}

}  // namespace lim
