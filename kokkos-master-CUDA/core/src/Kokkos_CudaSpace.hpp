/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CUDASPACE_HPP
#define KOKKOS_CUDASPACE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <Kokkos_Core_fwd.hpp>

#include <iosfwd>
#include <typeinfo>
#include <string>

#include <Kokkos_HostSpace.hpp>

#include <impl/Kokkos_Profiling_Interface.hpp>

#include <Cuda/Kokkos_Cuda_abort.hpp>

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
extern "C" bool kokkos_impl_cuda_pin_uvm_to_host();
extern "C" void kokkos_impl_cuda_set_pin_uvm_to_host(bool);
#endif

/*--------------------------------------------------------------------------*/

namespace Kokkos {

const int MAX_DEV = 12;

struct Kokkos_multi_dev_mem{
  float *space[12];
  int     num_dev;
};

template <typename T>
struct Kokkos_multi_dev_mem_test{
  T *array;
};

template <typename T>
struct Kokkos_multi_dev_mem_2{
  T *array;
};

typedef Kokkos_multi_dev_mem_2<float> ssKokkos_multi_dev_mem;
typedef Kokkos_multi_dev_mem_2<double> ddKokkos_multi_dev_mem;

struct sKokkos_multi_dev_mem{
  float* array;
};

struct dKokkos_multi_dev_mem{
  double* array;
};

/** \brief  Cuda on-device memory management */

class CudaSpace {
 public:
  //! Tag this class as a kokkos memory space
  using memory_space    = CudaSpace;
  using execution_space = Kokkos::Cuda;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using size_type = unsigned int;

  /*--------------------------------*/

  CudaSpace();
  CudaSpace(CudaSpace&& rhs)      = default;
  CudaSpace(const CudaSpace& rhs) = default;
  CudaSpace& operator=(CudaSpace&& rhs) = default;
  CudaSpace& operator=(const CudaSpace& rhs) = default;
  ~CudaSpace()                               = default;

  /**\brief  Allocate untracked memory in the cuda space */
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  template <class, class, class, class>
  friend class Kokkos::Experimental::LogicalMemorySpace;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  /*--------------------------------*/
  /** \brief  Error reporting for HostSpace attempt to access CudaSpace */
  static void access_error();
  static void access_error(const void* const);

 private:
  int m_device;  ///< Which Cuda device

  static constexpr const char* m_name = "Cuda";
  friend class Kokkos::Impl::SharedAllocationRecord<Kokkos::CudaSpace, void>;
};

namespace Impl {
/// \brief Initialize lock array for arbitrary size atomics.
///
/// Arbitrary atomics are implemented using a hash table of locks
/// where the hash value is derived from the address of the
/// object for which an atomic operation is performed.
/// This function initializes the locks to zero (unset).
void init_lock_arrays_cuda_space();

/// \brief Retrieve the pointer to the lock array for arbitrary size atomics.
///
/// Arbitrary atomics are implemented using a hash table of locks
/// where the hash value is derived from the address of the
/// object for which an atomic operation is performed.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* atomic_lock_array_cuda_space_ptr(bool deallocate = false);

/// \brief Retrieve the pointer to the scratch array for team and thread private
/// global memory.
///
/// Team and Thread private scratch allocations in
/// global memory are acquired via locks.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* scratch_lock_array_cuda_space_ptr(bool deallocate = false);

/// \brief Retrieve the pointer to the scratch array for unique identifiers.
///
/// Unique identifiers in the range 0-Cuda::concurrency
/// are provided via locks.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* threadid_lock_array_cuda_space_ptr(bool deallocate = false);
}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

/** \brief  Cuda memory that is accessible to Host execution space
 *          through Cuda's unified virtual memory (UVM) runtime.
 */
class CudaUVMSpace {
 public:
  //! Tag this class as a kokkos memory space
  using memory_space    = CudaUVMSpace;
  using execution_space = Cuda;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = unsigned int;

  /** \brief  If UVM capability is available */
  static bool available();

  /*--------------------------------*/
  /** \brief  CudaUVMSpace specific routine */
  KOKKOS_DEPRECATED static int number_of_allocations();

  /*--------------------------------*/

  /*--------------------------------*/

  CudaUVMSpace();
  CudaUVMSpace(CudaUVMSpace&& rhs)      = default;
  CudaUVMSpace(const CudaUVMSpace& rhs) = default;
  CudaUVMSpace& operator=(CudaUVMSpace&& rhs) = default;
  CudaUVMSpace& operator=(const CudaUVMSpace& rhs) = default;
  ~CudaUVMSpace()                                  = default;

  /**\brief  Allocate untracked memory in the cuda space */
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  template <class, class, class, class>
  friend class Kokkos::Experimental::LogicalMemorySpace;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
  static bool cuda_pin_uvm_to_host();
  static void cuda_set_pin_uvm_to_host(bool val);
#endif
  /*--------------------------------*/

 private:
  int m_device;  ///< Which Cuda device

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
  static bool kokkos_impl_cuda_pin_uvm_to_host_v;
#endif
  static constexpr const char* m_name = "CudaUVM";
};

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

/** \brief  Host memory that is accessible to Cuda execution space
 *          through Cuda's host-pinned memory allocation.
 */
class CudaHostPinnedSpace {
 public:
  //! Tag this class as a kokkos memory space
  /** \brief  Memory is in HostSpace so use the HostSpace::execution_space */
  using execution_space = HostSpace::execution_space;
  using memory_space    = CudaHostPinnedSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = unsigned int;

  /*--------------------------------*/

  CudaHostPinnedSpace();
  CudaHostPinnedSpace(CudaHostPinnedSpace&& rhs)      = default;
  CudaHostPinnedSpace(const CudaHostPinnedSpace& rhs) = default;
  CudaHostPinnedSpace& operator=(CudaHostPinnedSpace&& rhs) = default;
  CudaHostPinnedSpace& operator=(const CudaHostPinnedSpace& rhs) = default;
  ~CudaHostPinnedSpace()                                         = default;

  /**\brief  Allocate untracked memory in the space */
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  template <class, class, class, class>
  friend class Kokkos::Experimental::LogicalMemorySpace;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

 private:
  static constexpr const char* m_name = "CudaHostPinned";

  /*--------------------------------*/
};

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::CudaSpace,
                                              Kokkos::CudaSpace>::assignable,
              "");
static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::CudaUVMSpace,
                                              Kokkos::CudaUVMSpace>::assignable,
              "");
static_assert(
    Kokkos::Impl::MemorySpaceAccess<Kokkos::CudaHostPinnedSpace,
                                    Kokkos::CudaHostPinnedSpace>::assignable,
    "");

//----------------------------------------

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::CudaSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::CudaUVMSpace> {
  // HostSpace::execution_space != CudaUVMSpace::execution_space
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::CudaHostPinnedSpace> {
  // HostSpace::execution_space == CudaHostPinnedSpace::execution_space
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

//----------------------------------------

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace, Kokkos::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace, Kokkos::CudaUVMSpace> {
  // CudaSpace::execution_space == CudaUVMSpace::execution_space
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace, Kokkos::CudaHostPinnedSpace> {
  // CudaSpace::execution_space != CudaHostPinnedSpace::execution_space
  enum : bool { assignable = false };
  enum : bool { accessible = true };  // CudaSpace::execution_space
  enum : bool { deepcopy = true };
};

//----------------------------------------
// CudaUVMSpace::execution_space == Cuda
// CudaUVMSpace accessible to both Cuda and Host

template <>
struct MemorySpaceAccess<Kokkos::CudaUVMSpace, Kokkos::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };  // Cuda cannot access HostSpace
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaUVMSpace, Kokkos::CudaSpace> {
  // CudaUVMSpace::execution_space == CudaSpace::execution_space
  // Can access CudaUVMSpace from Host but cannot access CudaSpace from Host
  enum : bool { assignable = false };

  // CudaUVMSpace::execution_space can access CudaSpace
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaUVMSpace, Kokkos::CudaHostPinnedSpace> {
  // CudaUVMSpace::execution_space != CudaHostPinnedSpace::execution_space
  enum : bool { assignable = false };
  enum : bool { accessible = true };  // CudaUVMSpace::execution_space
  enum : bool { deepcopy = true };
};

//----------------------------------------
// CudaHostPinnedSpace::execution_space == HostSpace::execution_space
// CudaHostPinnedSpace accessible to both Cuda and Host

template <>
struct MemorySpaceAccess<Kokkos::CudaHostPinnedSpace, Kokkos::HostSpace> {
  enum : bool { assignable = false };  // Cannot access from Cuda
  enum : bool { accessible = true };   // CudaHostPinnedSpace::execution_space
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaHostPinnedSpace, Kokkos::CudaSpace> {
  enum : bool { assignable = false };  // Cannot access from Host
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaHostPinnedSpace, Kokkos::CudaUVMSpace> {
  enum : bool { assignable = false };  // different execution_space
  enum : bool { accessible = true };   // same accessibility
  enum : bool { deepcopy = true };
};

//----------------------------------------

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

void DeepCopyAsyncCuda(void* dst, const void* src, size_t n);

template <>
struct DeepCopy<CudaSpace, CudaSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Cuda&, void* dst, const void* src, size_t);
};

template <>
struct DeepCopy<CudaSpace, HostSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Cuda&, void* dst, const void* src, size_t);
};

template <>
struct DeepCopy<HostSpace, CudaSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Cuda&, void* dst, const void* src, size_t);
};

template <>
struct DeepCopy<CudaUVMSpace, CudaUVMSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaUVMSpace, HostSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, HostSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, HostSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<HostSpace, CudaUVMSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaHostPinnedSpace, CudaHostPinnedSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaHostPinnedSpace, HostSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, HostSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, HostSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<HostSpace, CudaHostPinnedSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaUVMSpace, CudaSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaSpace, CudaUVMSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaUVMSpace, CudaHostPinnedSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaHostPinnedSpace, CudaUVMSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaSpace, CudaHostPinnedSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <>
struct DeepCopy<CudaHostPinnedSpace, CudaSpace, Cuda> {
  DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(instance, dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaSpace, CudaSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaSpace, HostSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, HostSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<HostSpace, CudaSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, CudaSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaSpace, CudaUVMSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaSpace, CudaHostPinnedSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, HostSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaUVMSpace, CudaSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaUVMSpace, CudaUVMSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, CudaSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaUVMSpace, CudaHostPinnedSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, HostSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaUVMSpace, HostSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<CudaSpace, HostSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaHostPinnedSpace, CudaSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, CudaSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaHostPinnedSpace, CudaUVMSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, CudaSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaHostPinnedSpace, CudaHostPinnedSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, HostSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<CudaHostPinnedSpace, HostSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, HostSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<HostSpace, CudaUVMSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, CudaSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<HostSpace, CudaHostPinnedSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<HostSpace, HostSpace, Cuda>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence();
    DeepCopyAsyncCuda(dst, src, n);
  }
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in CudaSpace attempting to access HostSpace: error */
template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::CudaSpace,
                                           Kokkos::HostSpace> {
  enum : bool { value = false };
  KOKKOS_INLINE_FUNCTION static void verify(void) {
    Kokkos::abort("Cuda code attempted to access HostSpace memory");
  }

  KOKKOS_INLINE_FUNCTION static void verify(const void*) {
    Kokkos::abort("Cuda code attempted to access HostSpace memory");
  }
};

/** Running in CudaSpace accessing CudaUVMSpace: ok */
template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::CudaSpace,
                                           Kokkos::CudaUVMSpace> {
  enum : bool { value = true };
  KOKKOS_INLINE_FUNCTION static void verify(void) {}
  KOKKOS_INLINE_FUNCTION static void verify(const void*) {}
};

/** Running in CudaSpace accessing CudaHostPinnedSpace: ok */
template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::CudaSpace,
                                           Kokkos::CudaHostPinnedSpace> {
  enum : bool { value = true };
  KOKKOS_INLINE_FUNCTION static void verify(void) {}
  KOKKOS_INLINE_FUNCTION static void verify(const void*) {}
};

/** Running in CudaSpace attempting to access an unknown space: error */
template <class OtherSpace>
struct VerifyExecutionCanAccessMemorySpace<
    typename std::enable_if<!std::is_same<Kokkos::CudaSpace, OtherSpace>::value,
                            Kokkos::CudaSpace>::type,
    OtherSpace> {
  enum : bool { value = false };
  KOKKOS_INLINE_FUNCTION static void verify(void) {
    Kokkos::abort("Cuda code attempted to access unknown Space memory");
  }

  KOKKOS_INLINE_FUNCTION static void verify(const void*) {
    Kokkos::abort("Cuda code attempted to access unknown Space memory");
  }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access CudaSpace */
template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::HostSpace,
                                           Kokkos::CudaSpace> {
  enum : bool { value = false };
  inline static void verify(void) { CudaSpace::access_error(); }
  inline static void verify(const void* p) { CudaSpace::access_error(p); }
};

/** Running in HostSpace accessing CudaUVMSpace is OK */
template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::HostSpace,
                                           Kokkos::CudaUVMSpace> {
  enum : bool { value = true };
  inline static void verify(void) {}
  inline static void verify(const void*) {}
};

/** Running in HostSpace accessing CudaHostPinnedSpace is OK */
template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::HostSpace,
                                           Kokkos::CudaHostPinnedSpace> {
  enum : bool { value = true };
  KOKKOS_INLINE_FUNCTION static void verify(void) {}
  KOKKOS_INLINE_FUNCTION static void verify(const void*) {}
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <>
class SharedAllocationRecord<Kokkos::CudaSpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  friend class SharedAllocationRecord<Kokkos::CudaUVMSpace, void>;

  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase*);

  static ::cudaTextureObject_t attach_texture_object(
      const unsigned sizeof_alias, void* const alloc_ptr,
      const size_t alloc_size);

#ifdef KOKKOS_ENABLE_DEBUG
  static RecordBase s_root_record;
#endif

  ::cudaTextureObject_t m_tex_obj;
  const Kokkos::CudaSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_tex_obj(0), m_space() {}

  SharedAllocationRecord(
      const Kokkos::CudaSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  std::string get_label() const;

  static SharedAllocationRecord* allocate(const Kokkos::CudaSpace& arg_space,
                                          const std::string& arg_label,
                                          const size_t arg_alloc_size);

  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size);

  /**\brief  Allocate tracked memory in the space */
  static void allocate_tracked_multi_dev(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::Kokkos_multi_dev_mem* space,
                                const size_t arg_alloc_size);

  //template <typename T> static void allocate_tracked_multi_dev_test(const Kokkos::CudaSpace& arg_space,
  static void allocate_tracked_multi_dev_test(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos_multi_dev_mem_test<float>* (&space)[12],
                                //T* (&space)[12],
                                const size_t arg_alloc_size);

  static void allocate_tracked_multi_dev_test(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos_multi_dev_mem_test<double>* (&space)[12],
                                //T* (&space)[12],
                                const size_t arg_alloc_size);

  static void allocate_tracked_smulti_dev(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::sKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size);


  static void allocate_tracked_dmulti_dev(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::dKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size);
  
  static void allocate_tracked_multi_dev_stencil(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::Kokkos_multi_dev_mem* space,
                                const size_t arg_alloc_size,
                                int size_of_stencil);

  static void allocate_tracked_smulti_dev_stencil(const Kokkos::CudaSpace &arg_space,
                                const std::string &arg_alloc_label,
                                Kokkos::sKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size,
                                int size_of_stencil);

  static void allocate_tracked_dmulti_dev_stencil(const Kokkos::CudaSpace &arg_space,
                                const std::string &arg_alloc_label,
                                Kokkos::dKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size,
                                int size_of_stencil);

  static void allocate_tracked_ddmulti_dev_stencil(const Kokkos::CudaSpace &arg_space,
                                const std::string &arg_alloc_label,
                                Kokkos::ddKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size,
                                int size_of_stencil);

  static void allocate_tracked_multi_dev_stencil_auto(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::Kokkos_multi_dev_mem* space,
                                const size_t arg_alloc_size,
                                int size_of_stencil,
                                int num_dev);
  
  static void allocate_tracked_smulti_dev_stencil_auto(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::sKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size,
                                int size_of_stencil,
                                int num_dev);

  static void allocate_tracked_dmulti_dev_stencil_auto(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::dKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size,
                                int size_of_stencil,
                                int num_dev);

  static void multi_dev_stencil_ghost(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::Kokkos_multi_dev_mem* space,
                                int size_array,
                                int size_of_stencil);

  static void multi_dev_stencil_sghost(const Kokkos::CudaSpace &arg_space,
                                const std::string &arg_alloc_label,
                                Kokkos::sKokkos_multi_dev_mem* (&space)[12],
                                int size_array,
                                int size_of_stencil);

  static void multi_dev_stencil_dghost(const Kokkos::CudaSpace &arg_space,
                                const std::string &arg_alloc_label,
                                Kokkos::dKokkos_multi_dev_mem* (&space)[12],
                                int size_array,
                                int size_of_stencil);

  static void multi_dev_stencil_ghost_auto(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::Kokkos_multi_dev_mem* space,
                                int size_array,
                                int size_of_stencil,
                                int num_dev);

  static void multi_dev_stencil_sghost_auto(const Kokkos::CudaSpace &arg_space,
                                const std::string &arg_alloc_label,
                                Kokkos::sKokkos_multi_dev_mem* (&space)[12],
                                int size_array,
                                int size_of_stencil,
                                int num_dev);

  static void multi_dev_stencil_dghost_auto(const Kokkos::CudaSpace &arg_space,
                                const std::string &arg_alloc_label,
                                Kokkos::dKokkos_multi_dev_mem* (&space)[12],
                                int size_array,
                                int size_of_stencil,
                                int num_dev);

  static int multi_dev_auto(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size);
  
  static void allocate_tracked_multi_dev_auto(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::Kokkos_multi_dev_mem* space,
                                const size_t arg_alloc_size,
                                int num_dev);

  static void allocate_tracked_smulti_dev_auto(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::sKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size,
                                int num_dev);

  static void allocate_tracked_dmulti_dev_auto(const Kokkos::CudaSpace& arg_space,
                                const std::string& arg_label,
                                Kokkos::dKokkos_multi_dev_mem* (&space)[12],
                                const size_t arg_alloc_size,
                                int num_dev);

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  template <typename AliasType>
  inline ::cudaTextureObject_t attach_texture_object() {
    static_assert((std::is_same<AliasType, int>::value ||
                   std::is_same<AliasType, ::int2>::value ||
                   std::is_same<AliasType, ::int4>::value),
                  "Cuda texture fetch only supported for alias types of int, "
                  "::int2, or ::int4");

    if (m_tex_obj == 0) {
      m_tex_obj = attach_texture_object(sizeof(AliasType),
                                        (void*)RecordBase::m_alloc_ptr,
                                        RecordBase::m_alloc_size);
    }

    return m_tex_obj;
  }

  template <typename AliasType>
  inline int attach_texture_object_offset(const AliasType* const ptr) {
    // Texture object is attached to the entire allocation range
    return ptr - reinterpret_cast<AliasType*>(RecordBase::m_alloc_ptr);
  }

  static void print_records(std::ostream&, const Kokkos::CudaSpace&,
                            bool detail = false);
};

template <>
class SharedAllocationRecord<Kokkos::CudaUVMSpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase*);

  static RecordBase s_root_record;

  ::cudaTextureObject_t m_tex_obj;
  const Kokkos::CudaUVMSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_tex_obj(0), m_space() {}

  SharedAllocationRecord(
      const Kokkos::CudaUVMSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  std::string get_label() const;

  static SharedAllocationRecord* allocate(const Kokkos::CudaUVMSpace& arg_space,
                                          const std::string& arg_label,
                                          const size_t arg_alloc_size);

  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(const Kokkos::CudaUVMSpace& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size);

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  template <typename AliasType>
  inline ::cudaTextureObject_t attach_texture_object() {
    static_assert((std::is_same<AliasType, int>::value ||
                   std::is_same<AliasType, ::int2>::value ||
                   std::is_same<AliasType, ::int4>::value),
                  "Cuda texture fetch only supported for alias types of int, "
                  "::int2, or ::int4");

    if (m_tex_obj == 0) {
      m_tex_obj = SharedAllocationRecord<Kokkos::CudaSpace, void>::
          attach_texture_object(sizeof(AliasType),
                                (void*)RecordBase::m_alloc_ptr,
                                RecordBase::m_alloc_size);
    }

    return m_tex_obj;
  }

  template <typename AliasType>
  inline int attach_texture_object_offset(const AliasType* const ptr) {
    // Texture object is attached to the entire allocation range
    return ptr - reinterpret_cast<AliasType*>(RecordBase::m_alloc_ptr);
  }

  static void print_records(std::ostream&, const Kokkos::CudaUVMSpace&,
                            bool detail = false);
};

template <>
class SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase*);

  static RecordBase s_root_record;

  const Kokkos::CudaHostPinnedSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_space() {}

  SharedAllocationRecord(
      const Kokkos::CudaHostPinnedSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  std::string get_label() const;

  static SharedAllocationRecord* allocate(
      const Kokkos::CudaHostPinnedSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size);
  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(const Kokkos::CudaHostPinnedSpace& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size);

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  static void print_records(std::ostream&, const Kokkos::CudaHostPinnedSpace&,
                            bool detail = false);
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif /* #define KOKKOS_CUDASPACE_HPP */
