/*!
 * \file gradients.cpp
 * \brief Unit tests for gradient calculation.
 * \author P. Gomes, T. Albring
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

#include "catch.hpp"
#include "../../Common/include/geometry/CPhysicalGeometry.hpp"
#include "../../Common/include/containers/container_decorators.hpp"
#include "../../SU2_CFD/include/solvers/CSolver.hpp"
#include "../../SU2_CFD/include/gradients/computeGradientsGreenGauss.hpp"
#include "../../SU2_CFD/include/gradients/computeGradientsLeastSquares.hpp"

/*!
 * \brief Base class for gradient tests using a unit cube geometry.
 * Derived classes should implement operator (i,j), returning the value
 * of the test function, and method grad(i,j,k), returning the known
 * gradient.
 */
struct GradientTestBase {
  const string configOptions =
      "SOLVER= NAVIER_STOKES\n"
      "MESH_FORMAT= BOX\n"
      "INIT_OPTION= TD_CONDITIONS\n"
      "MARKER_HEATFLUX= (y_minus, 0.0, y_plus, 0.0)\n"
      "MARKER_FAR= (x_minus, x_plus, z_plus, z_minus)\n"
      "MESH_BOX_SIZE= 10,10,10\n"
      "MESH_BOX_LENGTH= 1,1,1\n"
      "MESH_BOX_OFFSET= 0,0,0\n";

  std::unique_ptr<CConfig> config;
  std::unique_ptr<CGeometry> geometry;

  GradientTestBase() {
    initConfig();
    initGeometry();
  }

  /*!
   * \brief Initialize the config structure
   */
  void initConfig() {
    auto origBuf = cout.rdbuf();
    cout.rdbuf(nullptr);
    stringstream ss(configOptions);
    config = std::unique_ptr<CConfig>(new CConfig(ss, SU2_COMPONENT::SU2_CFD, false));
    cout.rdbuf(origBuf);
  }

  /*!
   * \brief Initialize the geometry
   */
  void initGeometry() {
    auto origBuf = cout.rdbuf();
    cout.rdbuf(nullptr);
    {
      auto aux_geometry = std::unique_ptr<CGeometry>(new CPhysicalGeometry(config.get(), 0, 1));
      geometry = std::unique_ptr<CGeometry>(new CPhysicalGeometry(aux_geometry.get(), config.get()));
    }
    geometry->SetSendReceive(config.get());
    geometry->SetBoundaries(config.get());
    geometry->SetPoint_Connectivity();
    geometry->SetElement_Connectivity();
    geometry->SetBoundVolume();
    geometry->Check_IntElem_Orientation(config.get());
    geometry->Check_BoundElem_Orientation(config.get());
    geometry->SetEdges();
    geometry->SetVertex(config.get());
    geometry->SetControlVolume(config.get(), ALLOCATE);
    geometry->SetBoundControlVolume(config.get(), ALLOCATE);
    geometry->FindNormal_Neighbor(config.get());
    geometry->SetGlobal_to_Local_Point();
    geometry->PreprocessP2PComms(geometry.get(), config.get());

    cout.rdbuf(origBuf);
  }
};

struct LinearFunction : public GradientTestBase {
  const unsigned long nVar = 1;
  const su2double constant = -1.0;
  const su2double slope[3] = {1.0, 2.0, 3.0};

  /*!
   * \brief Return manufactured value.
   */
  su2double operator()(unsigned long iPoint, unsigned long) const {
    const auto coord = geometry->nodes->GetCoord(iPoint);
    return constant + GeometryToolbox::DotProduct(geometry->GetnDim(), slope, coord);
  }

  /*!
   * \brief Return reference value.
   */
  su2double grad(unsigned long, unsigned long, unsigned long iDim) const { return slope[iDim]; }
};

template <class T, class U>
void check(const T& ref, const U& calc, su2double tol = 1e-9) {
  su2double err = 0.0;
  for (auto iPoint = 0ul; iPoint < calc.length(); ++iPoint) {
    for (auto iVar = 0ul; iVar < calc.rows(); ++iVar)
      for (auto iDim = 0ul; iDim < calc.cols(); ++iDim)
        err = max(err, abs(calc(iPoint, iVar, iDim) - ref.grad(iPoint, iVar, iDim)));
  }
  CHECK(err < tol);
}

template <class TestField>
void testGreenGauss() {
  TestField field;
  C3DDoubleMatrix gradient(field.geometry->GetnPoint(), field.nVar, field.geometry->GetnDim());

  computeGradientsGreenGauss(nullptr, MPI_QUANTITIES::SOLUTION, PERIODIC_NONE, *field.geometry.get(),
                             *field.config.get(), field, 0, field.nVar, -1, gradient);
  check(field, gradient);
}

template <class TestField>
void testLeastSquares(bool weighted) {
  TestField field;
  const auto nDim = field.geometry->GetnDim();
  C3DDoubleMatrix R(field.geometry->GetnPoint(), nDim, nDim);
  C3DDoubleMatrix gradient(field.geometry->GetnPoint(), field.nVar, nDim);

  computeGradientsLeastSquares(nullptr, MPI_QUANTITIES::SOLUTION, PERIODIC_NONE, *field.geometry.get(),
                               *field.config.get(), weighted, field, 0, field.nVar, -1, gradient, R);
  check(field, gradient);
}

TEST_CASE("GG", "[Gradients]") { testGreenGauss<LinearFunction>(); }

TEST_CASE("LS", "[Gradients]") { testLeastSquares<LinearFunction>(false); }

TEST_CASE("WLS", "[Gradients]") { testLeastSquares<LinearFunction>(true); }

/*!
 * Check that the shared per-point function gives the same results with the host policy and accessor objects
 * (as used by computeGradientsGreenGauss) and with the policy and raw views intended for device code.
 */
TEST_CASE("GG point views", "[Gradients]") {
  LinearFunction field;
  constexpr size_t nDim = 3;
  const auto& geometry = *field.geometry;
  const size_t nPoint = geometry.GetnPoint();
  const size_t nPointDomain = geometry.GetnPointDomain();
  REQUIRE(geometry.GetnDim() == nDim);

  const auto geom = detail::makeGreenGaussGeomView(geometry);

  /*--- Values of the field stored in a plain array, read through a view. ---*/
  std::vector<su2double> values(nPoint);
  for (size_t iPoint = 0; iPoint < nPoint; ++iPoint) values[iPoint] = field(iPoint, 0);
  gg::FieldView<su2double> fieldView{values.data(), 1};

  C3DDoubleMatrix refGrad(nPoint, field.nVar, nDim);
  C3DDoubleMatrix viewGrad(nPoint, field.nVar, nDim);
  gg::GradView<su2double, nDim> gradView{viewGrad.data(), field.nVar};

  for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint) {
    gg::greenGaussPoint<nDim, su2double, detail::AdPreacc>(iPoint, geom, field, 0, field.nVar, refGrad);
    gg::greenGaussPoint<nDim, su2double, gg::NoPreacc>(iPoint, geom, fieldView, 0, field.nVar, gradView);
  }

  /*--- The two instantiations are separate pieces of code compiled with -ffast-math, so bitwise equality is not
   *    guaranteed (FMA contraction); the operation order is the same so the difference is only rounding. ---*/
  su2double err = 0.0;
  for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint)
    for (size_t iDim = 0; iDim < nDim; ++iDim)
      err = max(err, abs(refGrad(iPoint, 0, iDim) - viewGrad(iPoint, 0, iDim)));
  CHECK(err < 1e-12);
}
