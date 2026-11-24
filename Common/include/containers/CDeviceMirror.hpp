/*!
 * \file CDeviceMirror.hpp
 * \brief A GPU copy of a host array with explicit tracking of which copy is current.
 * \author SU2 Contributors
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

/*--- The class only exists in builds with CUDA, there is no stub: code that wants a mirror has to be
 * guarded by HAVE_CUDA, so using one in a build without CUDA is a compile error (undeclared name)
 * instead of something that silently does nothing. No CUDA header is included otherwise. ---*/
#ifdef HAVE_CUDA

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "../toolboxes/allocation_toolbox.hpp"

/*--- Stale-copy checks are on in debug builds, and can be forced on (or off, with
 * SU2_GPU_NO_CHECK_STALE) by the build. The setting must be the same in every translation unit
 * that includes this header, as it changes what the inline functions do. ---*/
#if !defined(SU2_GPU_CHECK_STALE) && !defined(SU2_GPU_NO_CHECK_STALE) && !defined(NDEBUG)
#define SU2_GPU_CHECK_STALE
#endif

namespace CDeviceMirrorDetail {

/*! \brief Called when a stale copy is about to be used, the message names the mirror and the access. */
using StaleHandler = void (*)(const char* message);

/*! \brief Default handler: print the message and abort. */
inline void AbortOnStale(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::abort();
}

/*! \brief The active handler. Unit tests replace it with one that throws, to be able to test the checks. */
inline StaleHandler& Handler() {
  static StaleHandler handler = &AbortOnStale;
  return handler;
}

}  // namespace CDeviceMirrorDetail

/*!
 * \class CDeviceMirror
 * \brief The GPU copy of one host array.
 * \details The host array is borrowed (it stays owned by whoever allocated it, e.g. a C2DContainer, which
 * is never touched by the mirror), the device buffer is owned and allocated once in Attach. The mirror tracks
 * which copy holds the current data:
 *  - Host: only the host copy is current, ToDevice() uploads.
 *  - Device: only the device copy is current (a kernel wrote it), ToHost() downloads.
 *  - Both: both are current, transfers are no-ops.
 * Whoever writes data must say so (HostModified after CPU writes, DeviceModified after a kernel writes),
 * the mirror cannot see it. This class is host-only: kernels take the raw pointers it hands out.
 * \note Not thread safe, one mirror is meant to be driven by one thread. Transfers are synchronous.
 * \tparam T - Trivially copyable element type.
 */
template <class T>
class CDeviceMirror {
  static_assert(std::is_trivially_copyable<T>::value, "A mirror can only hold trivially copyable types.");

 public:
  /*! \brief Which copy of the data is current. */
  enum class Valid : uint8_t { Host, Device, Both };

 private:
  T* host = nullptr;       /*!< \brief Borrowed host array. */
  size_t n = 0;            /*!< \brief Number of elements. */
  T* dev = nullptr;        /*!< \brief Owned device buffer. */
  Valid state = Valid::Host;
  bool pinned = false;     /*!< \brief Whether the host range is registered (page-locked) by this mirror. */
  std::string name;
  size_t nUp = 0, nDown = 0;  /*!< \brief Number of transfers performed, for diagnostics and tests. */

  /*! \brief Report a stale access through the handler. */
  void Stale(const char* what) const {
    const std::string msg = std::string("SU2 GPU stale-data error: ") + what + " on device mirror '" +
                            (name.empty() ? "<unnamed>" : name) + "' but the " +
                            (state == Valid::Host ? "host" : "device") + " copy is the current one.";
    CDeviceMirrorDetail::Handler()(msg.c_str());
  }

  /*! \brief Release the device buffer and the page-locking, leaves the mirror detached. */
  void Release() noexcept {
    if (dev != nullptr) (void)cudaFree(dev);
    if (pinned) GPUMemoryAllocation::pinned_unregister(host);
    host = nullptr;
    dev = nullptr;
    n = 0;
    pinned = false;
    state = Valid::Host;
  }

 public:
  CDeviceMirror() = default;

  /*! \brief Attach to a host array, see Attach. */
  CDeviceMirror(T* hostPtr, size_t numElements, const std::string& mirrorName = "") {
    Attach(hostPtr, numElements, mirrorName);
  }

  ~CDeviceMirror() { Release(); }

  /*--- A mirror owns device memory and a page-locking, copying would share them. ---*/
  CDeviceMirror(const CDeviceMirror&) = delete;
  CDeviceMirror& operator=(const CDeviceMirror&) = delete;

  CDeviceMirror(CDeviceMirror&& other) noexcept { *this = std::move(other); }

  CDeviceMirror& operator=(CDeviceMirror&& other) noexcept {
    if (this != &other) {
      Release();
      host = other.host;
      n = other.n;
      dev = other.dev;
      state = other.state;
      pinned = other.pinned;
      name = std::move(other.name);
      nUp = other.nUp;
      nDown = other.nDown;
      other.host = nullptr;
      other.dev = nullptr;
      other.n = 0;
      other.pinned = false;
      other.state = Valid::Host;
    }
    return *this;
  }

  /*!
   * \brief Attach the mirror to a host array and allocate its device buffer (once, nothing is reallocated
   * by the transfers). Attaching an already attached mirror first releases the old buffer.
   * The host range is page-locked if possible (never fatal, see GPUMemoryAllocation::pinned_register), it
   * must therefore stay allocated and not move for as long as the mirror is attached.
   * \note On entry the host copy is the current one. Throws std::runtime_error if the device is out of memory.
   * \param[in] hostPtr - First element of the host array.
   * \param[in] numElements - Number of elements.
   * \param[in] mirrorName - Optional name, used in the stale data messages.
   */
  void Attach(T* hostPtr, size_t numElements, const std::string& mirrorName = "") {
    Release();
    name = mirrorName;
    if (hostPtr == nullptr || numElements == 0) return;

    const auto err = cudaMalloc(reinterpret_cast<void**>(&dev), numElements * sizeof(T));
    if (err != cudaSuccess) {
      (void)cudaGetLastError();
      dev = nullptr;
      throw std::runtime_error("Could not allocate " + std::to_string(numElements * sizeof(T)) +
                               " bytes on the device for mirror '" + (name.empty() ? "<unnamed>" : name) +
                               "': " + cudaGetErrorString(err));
    }
    host = hostPtr;
    n = numElements;
    pinned = GPUMemoryAllocation::pinned_register(host, n * sizeof(T));
    state = Valid::Host;
    nUp = nDown = 0;
  }

  /*! \brief Release the device buffer and the page-locking, the mirror is detached afterwards. */
  void Detach() { Release(); }

  /*! \brief Upload the host data, only if the host copy is the only current one. Afterwards both are. */
  void ToDevice() {
    if (state != Valid::Host || dev == nullptr) return;
    gpuErrChk(cudaMemcpy(dev, host, n * sizeof(T), cudaMemcpyHostToDevice));
    ++nUp;
    state = Valid::Both;
  }

  /*! \brief Download the device data, only if the device copy is the only current one. Afterwards both are. */
  void ToHost() {
    if (state != Valid::Device || dev == nullptr) return;
    gpuErrChk(cudaMemcpy(host, dev, n * sizeof(T), cudaMemcpyDeviceToHost));
    ++nDown;
    state = Valid::Both;
  }

  /*! \brief Declare that the host data was written (call after CPU writes), the device copy is now stale. */
  void HostModified() {
    if (dev != nullptr) state = Valid::Host;
  }

  /*! \brief Declare that the device data was written (call after a kernel wrote it), the host copy is stale. */
  void DeviceModified() {
    if (dev != nullptr) state = Valid::Device;
  }

  /*! \brief Device pointer for a kernel that only reads. Checks that the device copy is current. */
  const T* DeviceRead() const {
#ifdef SU2_GPU_CHECK_STALE
    if (state == Valid::Host && dev != nullptr) Stale("DeviceRead");
#endif
    return dev;
  }

  /*!
   * \brief Device pointer for a kernel that writes. Marks the device copy as the current one, the caller is
   * expected to launch the kernel next. A kernel that also reads the old values must be preceded by ToDevice,
   * otherwise it reads whatever the device buffer holds (this cannot be checked, as no read is declared).
   */
  T* DeviceWrite() {
    if (dev != nullptr) state = Valid::Device;
    return dev;
  }

  /*! \brief Host pointer for reading on the CPU. Checks that the host copy is current. */
  const T* HostRead() const {
#ifdef SU2_GPU_CHECK_STALE
    if (state == Valid::Device) Stale("HostRead");
#endif
    return host;
  }

  /*! \brief The borrowed host pointer without any check (e.g. to write, then call HostModified). */
  T* HostData() { return host; }

  /*! \brief The device pointer without any check or state change. */
  T* DeviceData() { return dev; }

  Valid State() const { return state; }
  size_t Size() const { return n; }
  bool IsAttached() const { return dev != nullptr; }
  bool IsPinned() const { return pinned; }
  const std::string& Name() const { return name; }

  /*! \brief Number of host to device and device to host copies performed since the last Attach. */
  size_t NumUploads() const { return nUp; }
  size_t NumDownloads() const { return nDown; }
};

#endif  // HAVE_CUDA
