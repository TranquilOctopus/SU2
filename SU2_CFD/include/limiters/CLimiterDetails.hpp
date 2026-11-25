/*!
 * \file CLimiterDetails.hpp
 * \brief A class template that allows defining limiters via
 *        specialization of particular details.
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

#include "../../../Common/include/code_config.hpp"
#include "../../../Common/include/option_structure.hpp"
#include "../gradients/greenGaussPoint.hpp"

/*--- The functions that run on the GPU (limiterFunction, geometricFactor of the kinds that are offloaded, and the
 * small functions of LimiterHelpers) are marked SU2_HOST_DEVICE and are shared between the CPU and the CUDA
 * kernels (see limiterPoint.hpp). The nvcc compilation of the kernels includes this file without the host
 * infrastructure (CGeometry, CConfig, MPI, OpenMP), everything that needs it (the "preprocess" functions, the
 * global values of Venkatakrishnan-Wang, the kinds that are not offloaded) is excluded with __CUDACC__. ---*/

/*! \brief Maximum number of variables of the work arrays of the limiters (also the size of the per-variable
 *         parameters stored in the details classes, so that the classes can be passed to a kernel by value). */
constexpr size_t LIMITER_MAX_NVAR = 32;

/*!
 * \brief A traits class for limiters, see notes for "computeLimiters_impl()".
 * \ingroup FvmAlgos
 * \note There is no default implementation (the code will compile but not
 *       link) specialization is mandatory.
 */
template<LIMITER LimiterKind>
struct CLimiterDetails
{
  /*!
   * \brief Compute any global value that may be needed by the other functions.
   * \note This function is called once by multiple threads.
   */
  template<class FieldType>
  inline void preprocess(CGeometry&, CConfig&, size_t varBegin,
                         size_t varEnd, const FieldType&);

  /*!
   * \brief Geometric modifier (e.g. increase limiting near sharp edges).
   * \note This function is called once per point inside an AD pre-
   *       -accumulation region, newly used variables should be registered.
   */
  inline su2double geometricFactor(size_t iPoint, CGeometry&) const;

  /*!
   * \brief Smooth (usually) function of the maximum/minimum (positive/negative)
   *        gradient projections onto the edges, and the deltas over direct neighbors.
   *        Both proj and delta may be 0.0, beware of divisions.
   * \note This function is called twice (min/max) per point per variable
   *       (also inside an AD pre-accumulation region), the template parameter is the
   *       preaccumulation policy (see gg::NoPreacc) through which new inputs are registered.
   */
  template<class Preacc>
  inline su2double limiterFunction(size_t iVar, su2double proj, su2double delta) const;
};


/*!
 * \brief Common small functions used by limiters.
 * \ingroup FvmAlgos
 */
template<class Type = su2double>
struct LimiterHelpers
{
  FORCEINLINE static Type epsilon() {return std::numeric_limits<passivedouble>::epsilon();}

  /*! \brief MUSCL reconstruction increment using the displacement to the middle of an edge. */
  template <class Int>
  SU2_HOST_DEVICE FORCEINLINE static Type reconstructionIncrement(Int nDim, const Type* halfEdge, const Type* gradient,
                                                  const Type& value_i, const Type& value_j, const Type& kappa) {
    Type proj = 0.0;
    for (Int iDim = 0; iDim < nDim; ++iDim) proj += halfEdge[iDim] * gradient[iDim];
    const Type cent = 0.5 * (value_j - value_i);
    return umusclProjection(proj, cent, kappa);
  }

  SU2_HOST_DEVICE FORCEINLINE static Type umusclProjection(const Type& grad_proj, const Type& delta, const Type& kappa)
  {
    /*-------------------------------------------------------------------*/
    /*--- The MUSCL kappa-scheme reconstruction is typically written: ---*/
    /*---     V_L = V_i + 0.25 * dV_ij^kap, where                     ---*/
    /*---     dV_ij^kap = (1-kappa) dV_ij^upw + (1+kappa) dV_ij^cen,  ---*/
    /*---     dV_ij^cen = V_j - V_i,                                  ---*/
    /*---     dV_ij^upw = 2 grad(Vi) dot vector_ij - dV_ij^cen.       ---*/
    /*--- To maintain proper scaling for edge limiters, the result of ---*/
    /*--- this function is 0.5 * dV_ij^kap.                           ---*/
    /*-------------------------------------------------------------------*/
    const Type cent = 0.5 * delta;
    const Type upw = grad_proj - cent;
    return (1.0 - kappa) * upw + (1.0 + kappa) * cent;
  }

  SU2_HOST_DEVICE FORCEINLINE static Type venkatFunction(const Type& proj, const Type& delta, const Type& eps2)
  {
    Type y = delta*(delta+proj) + eps2;
    return (y + delta*proj) / (y + 2*proj*proj);
  }

  FORCEINLINE static Type vanAlbadaFunction(const Type& proj, const Type& delta, const Type& eps)
  {
    return delta * (proj + delta) / (pow(proj, 2) + pow(delta, 2) + eps);
  }

  FORCEINLINE static Type raisedSine(const Type& dist)
  {
    Type factor = 0.5*(1.0+dist+sin(PI_NUMBER*dist)/PI_NUMBER);
    return max(0.0, min(factor, 1.0));
  }

  SU2_HOST_DEVICE FORCEINLINE static Type r3Function(const Type& proj, const Type& delta, const Type& epsp)
  {
    Type Dp = fabs(delta);
    Type Dm = fabs(proj);
    if(Dp>(2.0*Dm)) return 1.0;
    Type y = pow(Dp, 3) + epsp;
    Type S3 = 4.0*Dm*Dm;
    return (y + Dp*S3) / (y + Dm*(delta*delta+S3));
  }

  SU2_HOST_DEVICE FORCEINLINE static Type r4Function(const Type& proj, const Type& delta, const Type& epsp)
  {
    Type Dp = fabs(delta);
    Type Dm = fabs(proj);
    if(Dp>(2.0*Dm)) return 1.0;
    Type y = pow(Dp, 4) + epsp;
    Type S4 = 2.0*Dm*(Dp*Dp-2.0*Dm*(Dp-2.0*Dm));
    return (y + Dp*S4) / (y + Dm*(pow(delta,3)+S4));
  }

  SU2_HOST_DEVICE FORCEINLINE static Type r5Function(const Type& proj, const Type& delta, const Type& epsp)
  {
    Type Dp = fabs(delta);
    Type Dm = fabs(proj);
    if(Dp>(2.0*Dm)) return 1.0;
    Type y = pow(Dp, 5) + epsp;
    Type S5 = 8.0*Dm*Dm*(Dp*Dp-2.0*Dm*(Dp-Dm));
    return (y + Dp*S5) / (y + Dm*(pow(delta,4)+S5));
  }
};


/*!
 * \brief Barth-Jespersen specialization.
 * \ingroup FvmAlgos
 */
template<>
struct CLimiterDetails<LIMITER::BARTH_JESPERSEN>
{
  su2double eps2;

  /*!
   * \brief Set a small epsilon to avoid divisions by 0.
   */
#ifndef __CUDACC__
  template<class... Ts>
  inline void preprocess(Ts&...) {eps2 = LimiterHelpers<>::epsilon();}
#endif

  /*!
   * \brief No geometric modification for this kind of limiter.
   */
  template<class... Ts>
  SU2_HOST_DEVICE inline su2double geometricFactor(Ts&...) const {return 1.0;}

  /*!
   * \brief Venkatakrishnan function with a numerical epsilon.
   */
  template<class Preacc>
  SU2_HOST_DEVICE inline su2double limiterFunction(size_t, su2double proj, su2double delta) const
  {
    return LimiterHelpers<>::venkatFunction(proj, delta, eps2);
  }
};


/*!
 * \brief Venkatakrishnan specialization.
 * \ingroup FvmAlgos
 */
template<>
struct CLimiterDetails<LIMITER::VENKATAKRISHNAN>
{
  su2double eps2;

  /*!
   * \brief Store the reference lenght based eps^2 parameter,
   *        limited to a small number to avoid divisions by 0.
   */
#ifndef __CUDACC__
  template<class... Ts>
  inline void preprocess(CGeometry&, const CConfig& config, Ts&...)
  {
    su2double L = config.GetRefElemLength();
    su2double K = config.GetVenkat_LimiterCoeff();
    su2double eps1 = fabs(L*K);
    eps2 = max(eps1*eps1*eps1, LimiterHelpers<>::epsilon());
  }
#endif

  /*!
   * \brief No geometric modification for this kind of limiter.
   */
  template<class... Ts>
  SU2_HOST_DEVICE inline su2double geometricFactor(Ts&...) const {return 1.0;}

  /*!
   * \brief Smooth function that disables limiting in smooth regions.
   */
  template<class Preacc>
  SU2_HOST_DEVICE inline su2double limiterFunction(size_t, su2double proj, su2double delta) const
  {
    return LimiterHelpers<>::venkatFunction(proj, delta, eps2);
  }
};


/*!
 * \brief Nishikawa's R3 limiter specialization.
 * \ingroup FvmAlgos
 */
template<>
struct CLimiterDetails<LIMITER::NISHIKAWA_R3>
{
  su2double epsp;

  /*!
   * \brief Store the reference lenght based eps^3 parameter,
   *        limited to a small number to avoid divisions by 0.
   */
#ifndef __CUDACC__
  template<class... Ts>
  inline void preprocess(CGeometry&, const CConfig& config, Ts&...)
  {
    su2double L = config.GetRefElemLength();
    su2double K = config.GetVenkat_LimiterCoeff();
    su2double eps1 = fabs(L*K);
    epsp = max(pow(eps1, 4), LimiterHelpers<>::epsilon());
  }
#endif

  /*!
   * \brief No geometric modification for this kind of limiter.
   */
  template<class... Ts>
  SU2_HOST_DEVICE inline su2double geometricFactor(Ts&...) const {return 1.0;}

  /*!
   * \brief Smooth function that disables limiting in smooth regions.
   */
  template<class Preacc>
  SU2_HOST_DEVICE inline su2double limiterFunction(size_t, su2double proj, su2double delta) const
  {
    return LimiterHelpers<>::r3Function(proj, delta, epsp);
  }
};


/*!
 * \brief Nishikawa's R4 limiter specialization.
 * \ingroup FvmAlgos
 */
template<>
struct CLimiterDetails<LIMITER::NISHIKAWA_R4>
{
  su2double epsp;

  /*!
   * \brief Store the reference lenght based eps^4 parameter,
   *        limited to a small number to avoid divisions by 0.
   */
#ifndef __CUDACC__
  template<class... Ts>
  inline void preprocess(CGeometry&, const CConfig& config, Ts&...)
  {
    su2double L = config.GetRefElemLength();
    su2double K = config.GetVenkat_LimiterCoeff();
    su2double eps1 = fabs(L*K);
    epsp = max(pow(eps1, 5), LimiterHelpers<>::epsilon());
  }
#endif

  /*!
   * \brief No geometric modification for this kind of limiter.
   */
  template<class... Ts>
  SU2_HOST_DEVICE inline su2double geometricFactor(Ts&...) const {return 1.0;}

  /*!
   * \brief Smooth function that disables limiting in smooth regions.
   */
  template<class Preacc>
  SU2_HOST_DEVICE inline su2double limiterFunction(size_t, su2double proj, su2double delta) const
  {
    return LimiterHelpers<>::r4Function(proj, delta, epsp);
  }
};


/*!
 * \brief Nishikawa's R5 limiter specialization.
 * \ingroup FvmAlgos
 */
template<>
struct CLimiterDetails<LIMITER::NISHIKAWA_R5>
{
  su2double epsp;

  /*!
   * \brief Store the reference lenght based eps^5 parameter,
   *        limited to a small number to avoid divisions by 0.
   */
#ifndef __CUDACC__
  template<class... Ts>
  inline void preprocess(CGeometry&, const CConfig& config, Ts&...)
  {
    su2double L = config.GetRefElemLength();
    su2double K = config.GetVenkat_LimiterCoeff();
    su2double eps1 = fabs(L*K);
    epsp = max(pow(eps1, 6), LimiterHelpers<>::epsilon());
  }
#endif

  /*!
   * \brief No geometric modification for this kind of limiter.
   */
  template<class... Ts>
  SU2_HOST_DEVICE inline su2double geometricFactor(Ts&...) const {return 1.0;}

  /*!
   * \brief Smooth function that disables limiting in smooth regions.
   */
  template<class Preacc>
  SU2_HOST_DEVICE inline su2double limiterFunction(size_t, su2double proj, su2double delta) const
  {
    return LimiterHelpers<>::r5Function(proj, delta, epsp);
  }
};

/*!
 * \brief Venkatakrishnan-Wang specialization.
 * \ingroup FvmAlgos
 * \note eps2 is a plain array (not a container) so that the object can be copied to a kernel.
 */
template<>
struct CLimiterDetails<LIMITER::VENKATAKRISHNAN_WANG>
{
#ifndef __CUDACC__
  static su2activevector sharedMin, sharedMax;
#endif
  su2double eps2[LIMITER_MAX_NVAR];

#ifndef __CUDACC__
  /*!
   * \brief Store the solution range based eps^2 parameter.
   */
  template<class FieldType>
  inline void preprocess(CGeometry& geometry, const CConfig& config, size_t varBegin,
                         size_t varEnd, const FieldType& field)
  {
    /*--- Determine the max and min global value for each variable. ---*/

    su2double largeNum = 0.1*std::numeric_limits<passivedouble>::max();

    /*--- Allocate the static members (shared between threads) to
     * perform the reduction across all threads in the rank. ---*/

    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS
    {
      sharedMin.resize(varEnd) = largeNum;
      sharedMax.resize(varEnd) =-largeNum;
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS

    /*--- Per thread reduction. ---*/

    su2activevector localMin(varEnd), localMax(varEnd);
    localMin = largeNum;
    localMax =-largeNum;

    SU2_OMP_FOR_(schedule(static, 512) SU2_NOWAIT)
    for(size_t iPoint = 0; iPoint < geometry.GetnPointDomain(); ++iPoint)
    {
      for(size_t iVar = varBegin; iVar < varEnd; ++iVar)
      {
        localMin(iVar) = min(localMin(iVar), field(iPoint, iVar));
        localMax(iVar) = max(localMax(iVar), field(iPoint, iVar));
      }
    }
    END_SU2_OMP_FOR

    /*--- Per rank reduction. ---*/

    for(size_t iVar = varBegin; iVar < varEnd; ++iVar)
    {
      atomicMin(localMin(iVar), sharedMin(iVar));
      atomicMax(localMax(iVar), sharedMax(iVar));
    }

    /*--- Global reduction. ---*/

    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS
    {
      localMin = sharedMin;
      SU2_MPI::Allreduce(localMin.data(), sharedMin.data(), varEnd, MPI_DOUBLE, MPI_MIN, SU2_MPI::GetComm());

      localMax = sharedMax;
      SU2_MPI::Allreduce(localMax.data(), sharedMax.data(), varEnd, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS

    /*--- Compute eps^2 (each thread has its own copy of it). ---*/

    setEps2(config, varBegin, varEnd, sharedMin, sharedMax);
  }

  /*!
   * \brief Compute eps^2 from the global (all ranks) minimum and maximum of each variable, the last step of
   *        "preprocess", also used when the extrema come from the GPU.
   * \param[in] globalMin - Object implementing operator() (iVar).
   * \param[in] globalMax - As above, maximum.
   */
  template<class Vector>
  inline void setEps2(const CConfig& config, size_t varBegin, size_t varEnd,
                      const Vector& globalMin, const Vector& globalMax)
  {
    su2double K = config.GetVenkat_LimiterCoeff();

    for(size_t iVar = varBegin; iVar < varEnd; ++iVar)
    {
      su2double range = globalMax(iVar) - globalMin(iVar);
      eps2[iVar] = max(pow(K*range, 2), LimiterHelpers<>::epsilon());
    }
  }
#endif

  /*!
   * \brief No geometric modification for this kind of limiter.
   */
  template<class... Ts>
  SU2_HOST_DEVICE inline su2double geometricFactor(Ts&...) const {return 1.0;}

  /*!
   * \brief Smooth function that disables limiting in smooth regions.
   */
  template<class Preacc>
  SU2_HOST_DEVICE inline su2double limiterFunction(size_t iVar, su2double proj, su2double delta) const
  {
    Preacc::In(eps2[iVar]);
    return LimiterHelpers<>::venkatFunction(proj, delta, eps2[iVar]);
  }
};


#ifndef __CUDACC__

/*!
 * \brief Venkatakrishnan with sharp edge modification.
 * \ingroup FvmAlgos
 */
template<>
struct CLimiterDetails<LIMITER::SHARP_EDGES>
{
  su2double eps1, eps2, sharpCoeff;

  /*!
   * \brief Store the reference lenght based eps^2 parameter.
   */
  template<class... Ts>
  inline void preprocess(CGeometry&, const CConfig& config, Ts&...)
  {
    sharpCoeff = config.GetAdjSharp_LimiterCoeff();
    su2double L = config.GetRefElemLength();
    su2double K = config.GetVenkat_LimiterCoeff();
    eps1 = fabs(L*K);
    eps2 = max(eps1*eps1*eps1, LimiterHelpers<>::epsilon());
  }

  /*!
   * \brief Full limiting (1st order) near sharp edges.
   */
  inline su2double geometricFactor(size_t iPoint, CGeometry& geometry) const
  {
    AD::SetPreaccIn(geometry.nodes->GetSharpEdge_Distance(iPoint));
    su2double dist = geometry.nodes->GetSharpEdge_Distance(iPoint)/(sharpCoeff*eps1)-1.0;
    return LimiterHelpers<>::raisedSine(dist);
  }

  /*!
   * \brief Smooth function that disables limiting in smooth regions.
   */
  template<class Preacc>
  inline su2double limiterFunction(size_t, su2double proj, su2double delta) const
  {
    return LimiterHelpers<>::venkatFunction(proj, delta, eps2);
  }
};


/*!
 * \brief Venkatakrishnan with wall distance modification.
 * \ingroup FvmAlgos
 */
template<>
struct CLimiterDetails<LIMITER::WALL_DISTANCE>
{
  su2double eps1, eps2, sharpCoeff;

  /*!
   * \brief Store the reference lenght based eps^2 parameter.
   */
  template<class... Ts>
  inline void preprocess(CGeometry&, const CConfig& config, Ts&...)
  {
    sharpCoeff = config.GetAdjSharp_LimiterCoeff();
    su2double L = config.GetRefElemLength();
    su2double K = config.GetVenkat_LimiterCoeff();
    eps1 = fabs(L*K);
    eps2 = max(eps1*eps1*eps1, LimiterHelpers<>::epsilon());
  }

  /*!
   * \brief Full limiting (1st order) near walls.
   */
  inline su2double geometricFactor(size_t iPoint, CGeometry& geometry) const
  {
    AD::SetPreaccIn(geometry.nodes->GetWall_Distance(iPoint));
    su2double dist = geometry.nodes->GetWall_Distance(iPoint)/(sharpCoeff*eps1)-1.0;
    return LimiterHelpers<>::raisedSine(dist);
  }

  /*!
   * \brief Smooth function that disables limiting in smooth regions.
   */
  template<class Preacc>
  inline su2double limiterFunction(size_t, su2double proj, su2double delta) const
  {
    return LimiterHelpers<>::venkatFunction(proj, delta, eps2);
  }
};

#endif  // __CUDACC__
