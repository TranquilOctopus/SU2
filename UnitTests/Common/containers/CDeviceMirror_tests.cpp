/*!
 * \file CDeviceMirror_tests.cpp
 * \brief Unit tests for CDeviceMirror and SU2_HOST_DEVICE.
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

#include "../../../Common/include/code_config.hpp"

/*--- The tests need a device, they are only part of builds with the CUDA kernels. ---*/
#ifdef SU2_ENABLE_CUDA_KERNELS

#include <string>
#include <vector>

#include "../../../Common/include/containers/CDeviceMirror.hpp"
#include "CDeviceMirror_kernels.hpp"

namespace {

using Mirror = CDeviceMirror<double>;
using Valid = Mirror::Valid;

/*--- Copy a device array to a vector without going through the mirror, to look at what is really there. ---*/
std::vector<double> Peek(const double* devPtr, size_t n) {
  std::vector<double> out(n);
  REQUIRE(cudaMemcpy(out.data(), devPtr, n * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);
  return out;
}

size_t FreeDeviceMemory() {
  size_t freeBytes = 0, totalBytes = 0;
  REQUIRE(cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess);
  return freeBytes;
}

/*--- Test replacement of the stale data handler: throw instead of abort. ---*/
struct StaleError {
  std::string message;
};
void ThrowOnStale(const char* message) { throw StaleError{message}; }

struct ScopedThrowingHandler {
  CDeviceMirrorDetail::StaleHandler old = CDeviceMirrorDetail::Handler();
  ScopedThrowingHandler() { CDeviceMirrorDetail::Handler() = &ThrowOnStale; }
  ~ScopedThrowingHandler() { CDeviceMirrorDetail::Handler() = old; }
};

}  // namespace

TEST_CASE("SU2_HOST_DEVICE", "[CUDA]") {
  /*--- The function works on both sides, and the same macro is empty for the host compiler. ---*/
  CHECK(CDeviceMirrorTestTwiceOnHost(21.0) == 42.0);
  std::vector<double> x{1, 2, 3};
  Mirror m(x.data(), x.size(), "x");
  m.ToDevice();
  CDeviceMirrorTestDouble(m.DeviceWrite(), x.size());
  CHECK(Peek(m.DeviceRead(), 3) == std::vector<double>{2, 4, 6});
}

TEST_CASE("CDeviceMirror round trip", "[CUDA]") {
  const size_t n = 1000;
  std::vector<double> data(n);
  for (size_t i = 0; i < n; ++i) data[i] = double(i) + 0.5;

  Mirror m;
  CHECK_FALSE(m.IsAttached());
  m.Attach(data.data(), n, "data");
  REQUIRE(m.IsAttached());
  CHECK(m.Size() == n);
  CHECK(m.State() == Valid::Host);

  m.ToDevice();
  CHECK(m.State() == Valid::Both);
  CDeviceMirrorTestDouble(m.DeviceWrite(), n);
  CHECK(m.State() == Valid::Device);
  m.DeviceModified();
  CHECK(m.State() == Valid::Device);
  m.ToHost();
  CHECK(m.State() == Valid::Both);

  for (size_t i = 0; i < n; ++i) REQUIRE(data[i] == 2 * (double(i) + 0.5));
  CHECK(m.HostRead() == data.data());
}

TEST_CASE("CDeviceMirror does no redundant copies", "[CUDA]") {
  std::vector<double> data(64, 1.0);
  Mirror m(data.data(), data.size(), "data");

  /*--- Nothing to download while the host copy is current. ---*/
  m.ToHost();
  CHECK(m.NumDownloads() == 0);
  CHECK(m.State() == Valid::Host);

  m.ToDevice();
  m.ToDevice();
  m.ToDevice();
  CHECK(m.NumUploads() == 1);
  CHECK(m.State() == Valid::Both);

  /*--- Nothing to download while both are current. ---*/
  m.ToHost();
  CHECK(m.NumDownloads() == 0);

  CDeviceMirrorTestDouble(m.DeviceWrite(), data.size());
  m.ToHost();
  m.ToHost();
  CHECK(m.NumDownloads() == 1);
  CHECK(m.NumUploads() == 1);

  /*--- Nothing to upload after a download, both are current. ---*/
  m.ToDevice();
  CHECK(m.NumUploads() == 1);
}

TEST_CASE("CDeviceMirror HostModified re-uploads", "[CUDA]") {
  std::vector<double> data(16, 1.0);
  Mirror m(data.data(), data.size(), "data");
  m.ToDevice();
  REQUIRE(m.NumUploads() == 1);

  for (auto& x : data) x = 5.0;
  m.HostModified();
  CHECK(m.State() == Valid::Host);
  m.ToDevice();
  CHECK(m.NumUploads() == 2);
  CHECK(m.State() == Valid::Both);
  CHECK(Peek(m.DeviceRead(), data.size()) == std::vector<double>(16, 5.0));

  /*--- Without HostModified the mirror cannot know, and does not copy. ---*/
  for (auto& x : data) x = 7.0;
  m.ToDevice();
  CHECK(m.NumUploads() == 2);
  CHECK(Peek(m.DeviceRead(), data.size()) == std::vector<double>(16, 5.0));
}

TEST_CASE("CDeviceMirror memory management", "[CUDA]") {
  /*--- 64 MB mirrors: a leak per cycle would add up to many GB, which the (noisy, shared) free memory
   * query can reliably tell apart from other processes using the device. ---*/
  const size_t n = size_t(8) << 20;
  const size_t mb = size_t(1) << 20;
  std::vector<double> a(n, 1.0), b(n, 2.0);
  const size_t freeBefore = FreeDeviceMemory();

  {
    Mirror m;
    for (int cycle = 0; cycle < 40; ++cycle) {
      /*--- Alternate hosts: re-attaching must release the previous buffer. ---*/
      m.Attach(cycle % 2 ? a.data() : b.data(), n, "big");
      m.ToDevice();
    }
    const size_t freeDuring = FreeDeviceMemory();
    CHECK(freeBefore > freeDuring);  // the buffer is really there
    CHECK(m.NumUploads() == 1);      // counters restart on Attach
  }

  /*--- Destructor ran: at most a few hundred MB of noise, versus 40 * 64 MB = 2.5 GB if buffers leaked. ---*/
  const size_t freeAfter = FreeDeviceMemory();
  const size_t lost = freeBefore > freeAfter ? freeBefore - freeAfter : 0;
  CHECK(lost < 1000 * mb);

  /*--- The host ranges were unregistered, registering them again works (not the case if still registered). ---*/
  CHECK(cudaHostRegister(a.data(), n * sizeof(double), cudaHostRegisterDefault) == cudaSuccess);
  CHECK(cudaHostUnregister(a.data()) == cudaSuccess);
  CHECK(cudaHostRegister(b.data(), n * sizeof(double), cudaHostRegisterDefault) == cudaSuccess);
  CHECK(cudaHostUnregister(b.data()) == cudaSuccess);
}

TEST_CASE("CDeviceMirror is movable", "[CUDA]") {
  std::vector<double> data{1, 2, 3, 4};
  static_assert(!std::is_copy_constructible<Mirror>::value, "mirrors must not be copyable");
  static_assert(!std::is_copy_assignable<Mirror>::value, "mirrors must not be copyable");
  static_assert(std::is_nothrow_move_constructible<Mirror>::value, "mirrors must be movable");

  Mirror a(data.data(), data.size(), "a");
  a.ToDevice();
  const double* dev = a.DeviceRead();

  Mirror b(std::move(a));
  CHECK_FALSE(a.IsAttached());
  CHECK(b.IsAttached());
  CHECK(b.DeviceRead() == dev);  // same buffer, no new allocation
  CHECK(b.State() == Valid::Both);
  CHECK(b.Name() == "a");

  Mirror c;
  c = std::move(b);
  CHECK_FALSE(b.IsAttached());
  CHECK(c.DeviceRead() == dev);
  CHECK(c.NumUploads() == 1);
}

TEST_CASE("CDeviceMirror stale data detection", "[CUDA]") {
  /*--- The abort cannot be tested directly (it ends the test process), so the handler that the checks
   * call is replaced by one that throws. This is the code path of the default handler up to std::abort. ---*/
#ifdef SU2_GPU_CHECK_STALE
  ScopedThrowingHandler guard;
  std::vector<double> data(8, 1.0);
  Mirror m(data.data(), data.size(), "velocity");

  /*--- Host is current: the device copy is stale. ---*/
  REQUIRE(m.State() == Valid::Host);
  try {
    m.DeviceRead();
    FAIL("DeviceRead on a stale device copy was not detected");
  } catch (const StaleError& e) {
    CHECK(e.message.find("velocity") != std::string::npos);
    CHECK(e.message.find("DeviceRead") != std::string::npos);
  }
  CHECK_NOTHROW(m.HostRead());

  m.ToDevice();
  CHECK_NOTHROW(m.DeviceRead());
  CHECK_NOTHROW(m.HostRead());

  /*--- Device is current: the host copy is stale. ---*/
  CDeviceMirrorTestDouble(m.DeviceWrite(), data.size());
  REQUIRE(m.State() == Valid::Device);
  try {
    m.HostRead();
    FAIL("HostRead on a stale host copy was not detected");
  } catch (const StaleError& e) {
    CHECK(e.message.find("velocity") != std::string::npos);
    CHECK(e.message.find("HostRead") != std::string::npos);
  }
  CHECK_NOTHROW(m.DeviceRead());

  m.ToHost();
  CHECK_NOTHROW(m.HostRead());
  CHECK(data[3] == 2.0);

  /*--- After HostModified the device copy is stale again. ---*/
  m.HostModified();
  CHECK_THROWS_AS(m.DeviceRead(), StaleError);

  /*--- An unnamed mirror says so. ---*/
  std::vector<double> other(8, 1.0);  // not the range of m, it would overlap its page-locking
  Mirror u(other.data(), other.size());
  try {
    u.DeviceRead();
    FAIL("not detected");
  } catch (const StaleError& e) {
    CHECK(e.message.find("<unnamed>") != std::string::npos);
  }
#else
  /*--- Without the checks only the state logic is available. ---*/
  std::vector<double> data(8, 1.0);
  Mirror m(data.data(), data.size());
  CHECK(m.State() == Valid::Host);
  CHECK_NOTHROW(m.DeviceRead());
#endif
}

#endif  // SU2_ENABLE_CUDA_KERNELS
