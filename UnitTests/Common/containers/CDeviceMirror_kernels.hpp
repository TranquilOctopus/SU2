/*!
 * \file CDeviceMirror_kernels.hpp
 * \brief Declarations of the (nvcc compiled) helpers used by the CDeviceMirror unit tests.
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

/*! \brief Multiply every entry of a device array by two, using a function marked SU2_HOST_DEVICE. Synchronizes. */
void CDeviceMirrorTestDouble(double* devPtr, size_t n);

/*! \brief The same SU2_HOST_DEVICE function, called on the host from the nvcc compiled file. */
double CDeviceMirrorTestTwiceOnHost(double x);
