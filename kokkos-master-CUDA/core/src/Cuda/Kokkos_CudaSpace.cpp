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

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <atomic>

#include <Kokkos_Core.hpp>
#include <Kokkos_Cuda.hpp>
#include <Kokkos_CudaSpace.hpp>
#include <omp.h>

//#include <Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_MemorySpace.hpp>

#include <impl/Kokkos_Tools.hpp>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

//static float*  sarray_pointers[Kokkos::MAX_DEV];
//static double* darray_pointers[Kokkos::MAX_DEV];

namespace Impl {

namespace {

static std::atomic<int> num_uvm_allocations(0);

cudaStream_t get_deep_copy_stream() {
  static cudaStream_t s = nullptr;
  if (s == nullptr) {
    cudaStreamCreate(&s);
  }
  return s;
}
}  // namespace

DeepCopy<CudaSpace, CudaSpace, Cuda>::DeepCopy(void *dst, const void *src,
                                               size_t n) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, n, cudaMemcpyDefault));
}

DeepCopy<HostSpace, CudaSpace, Cuda>::DeepCopy(void *dst, const void *src,
                                               size_t n) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, n, cudaMemcpyDefault));
}

DeepCopy<CudaSpace, HostSpace, Cuda>::DeepCopy(void *dst, const void *src,
                                               size_t n) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, n, cudaMemcpyDefault));
}

DeepCopy<CudaSpace, CudaSpace, Cuda>::DeepCopy(const Cuda &instance, void *dst,
                                               const void *src, size_t n) {
  CUDA_SAFE_CALL(
      cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, instance.cuda_stream()));
}

DeepCopy<HostSpace, CudaSpace, Cuda>::DeepCopy(const Cuda &instance, void *dst,
                                               const void *src, size_t n) {
  CUDA_SAFE_CALL(
      cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, instance.cuda_stream()));
}

DeepCopy<CudaSpace, HostSpace, Cuda>::DeepCopy(const Cuda &instance, void *dst,
                                               const void *src, size_t n) {
  CUDA_SAFE_CALL(
      cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, instance.cuda_stream()));
}

void DeepCopyAsyncCuda(void *dst, const void *src, size_t n) {
  cudaStream_t s = get_deep_copy_stream();
  CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, s));
  cudaStreamSynchronize(s);
}

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

void CudaSpace::access_error() {
  const std::string msg(
      "Kokkos::CudaSpace::access_error attempt to execute Cuda function from "
      "non-Cuda space");
  Kokkos::Impl::throw_runtime_exception(msg);
}

void CudaSpace::access_error(const void *const) {
  const std::string msg(
      "Kokkos::CudaSpace::access_error attempt to execute Cuda function from "
      "non-Cuda space");
  Kokkos::Impl::throw_runtime_exception(msg);
}

/*--------------------------------------------------------------------------*/

bool CudaUVMSpace::available() {
#if defined(CUDA_VERSION) && !defined(__APPLE__)
  enum : bool { UVM_available = true };
#else
  enum : bool { UVM_available = false };
#endif
  return UVM_available;
}

/*--------------------------------------------------------------------------*/

int CudaUVMSpace::number_of_allocations() {
  return Kokkos::Impl::num_uvm_allocations.load();
}
#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
// The purpose of the following variable is to allow a state-based choice
// for pinning UVM allocations to the CPU. For now this is considered
// an experimental debugging capability - with the potential to work around
// some CUDA issues.
bool CudaUVMSpace::kokkos_impl_cuda_pin_uvm_to_host_v = false;

bool CudaUVMSpace::cuda_pin_uvm_to_host() {
  return CudaUVMSpace::kokkos_impl_cuda_pin_uvm_to_host_v;
}
void CudaUVMSpace::cuda_set_pin_uvm_to_host(bool val) {
  CudaUVMSpace::kokkos_impl_cuda_pin_uvm_to_host_v = val;
}
#endif
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
bool kokkos_impl_cuda_pin_uvm_to_host() {
  return Kokkos::CudaUVMSpace::cuda_pin_uvm_to_host();
}

void kokkos_impl_cuda_set_pin_uvm_to_host(bool val) {
  Kokkos::CudaUVMSpace::cuda_set_pin_uvm_to_host(val);
}
#endif

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

CudaSpace::CudaSpace() : m_device(Kokkos::Cuda().cuda_device()) {}

CudaUVMSpace::CudaUVMSpace() : m_device(Kokkos::Cuda().cuda_device()) {}

CudaHostPinnedSpace::CudaHostPinnedSpace() {}

//==============================================================================
// <editor-fold desc="allocate()"> {{{1

void *CudaSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *CudaSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                          const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void *CudaSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  void *ptr = nullptr;

  //printf("Allocating memory (%d bytes) in GPU\n", arg_alloc_size);
  auto error_code = cudaMalloc(&ptr, arg_alloc_size);
  if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
    cudaGetLastError();  // This is the only way to clear the last error, which
                         // we should do here since we're turning it into an
                         // exception here
    throw Experimental::CudaRawMemoryAllocationFailure(
        arg_alloc_size, error_code,
        Experimental::RawMemoryAllocationFailure::AllocationMechanism::
            CudaMalloc);
  }

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}

void *CudaUVMSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void *CudaUVMSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                             const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void *CudaUVMSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  void *ptr = nullptr;

  Cuda::impl_static_fence();
  if (arg_alloc_size > 0) {
    Kokkos::Impl::num_uvm_allocations++;

    auto error_code =
        cudaMallocManaged(&ptr, arg_alloc_size, cudaMemAttachGlobal);

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
    if (Kokkos::CudaUVMSpace::cuda_pin_uvm_to_host())
      cudaMemAdvise(ptr, arg_alloc_size, cudaMemAdviseSetPreferredLocation,
                    cudaCpuDeviceId);
#endif

    if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
      cudaGetLastError();  // This is the only way to clear the last error,
                           // which we should do here since we're turning it
                           // into an exception here
      throw Experimental::CudaRawMemoryAllocationFailure(
          arg_alloc_size, error_code,
          Experimental::RawMemoryAllocationFailure::AllocationMechanism::
              CudaMallocManaged);
    }
  }
  Cuda::impl_static_fence();
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}
void *CudaHostPinnedSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void *CudaHostPinnedSpace::allocate(const char *arg_label,
                                    const size_t arg_alloc_size,
                                    const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void *CudaHostPinnedSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  void *ptr = nullptr;

  auto error_code = cudaHostAlloc(&ptr, arg_alloc_size, cudaHostAllocDefault);
  if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
    cudaGetLastError();  // This is the only way to clear the last error, which
                         // we should do here since we're turning it into an
                         // exception here
    throw Experimental::CudaRawMemoryAllocationFailure(
        arg_alloc_size, error_code,
        Experimental::RawMemoryAllocationFailure::AllocationMechanism::
            CudaHostAlloc);
  }
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}

// </editor-fold> end allocate() }}}1
//==============================================================================
void CudaSpace::deallocate(void *const arg_alloc_ptr,
                           const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void CudaSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                           const size_t arg_alloc_size,
                           const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void CudaSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }

  try {
    CUDA_SAFE_CALL(cudaFree(arg_alloc_ptr));
  } catch (...) {
  }
}
void CudaUVMSpace::deallocate(void *const arg_alloc_ptr,
                              const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void CudaUVMSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                              const size_t arg_alloc_size

                              ,
                              const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void CudaUVMSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size

    ,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  Cuda::impl_static_fence();
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  try {
    if (arg_alloc_ptr != nullptr) {
      Kokkos::Impl::num_uvm_allocations--;
      CUDA_SAFE_CALL(cudaFree(arg_alloc_ptr));
    }
  } catch (...) {
  }
  Cuda::impl_static_fence();
}

void CudaHostPinnedSpace::deallocate(void *const arg_alloc_ptr,
                                     const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void CudaHostPinnedSpace::deallocate(const char *arg_label,
                                     void *const arg_alloc_ptr,
                                     const size_t arg_alloc_size,
                                     const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void CudaHostPinnedSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  try {
    CUDA_SAFE_CALL(cudaFreeHost(arg_alloc_ptr));
  } catch (...) {
  }
}

}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_ENABLE_DEBUG
SharedAllocationRecord<void, void>
    SharedAllocationRecord<Kokkos::CudaSpace, void>::s_root_record;

SharedAllocationRecord<void, void>
    SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::s_root_record;

SharedAllocationRecord<void, void>
    SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::s_root_record;
#endif

::cudaTextureObject_t
SharedAllocationRecord<Kokkos::CudaSpace, void>::attach_texture_object(
    const unsigned sizeof_alias, void *const alloc_ptr,
    size_t const alloc_size) {
  enum { TEXTURE_BOUND_1D = 1u << 27 };

  if ((alloc_ptr == nullptr) ||
      (sizeof_alias * TEXTURE_BOUND_1D <= alloc_size)) {
    std::ostringstream msg;
    msg << "Kokkos::CudaSpace ERROR: Cannot attach texture object to"
        << " alloc_ptr(" << alloc_ptr << ")"
        << " alloc_size(" << alloc_size << ")"
        << " max_size(" << (sizeof_alias * TEXTURE_BOUND_1D) << ")";
    std::cerr << msg.str() << std::endl;
    std::cerr.flush();
    Kokkos::Impl::throw_runtime_exception(msg.str());
  }

  ::cudaTextureObject_t tex_obj;

  struct cudaResourceDesc resDesc;
  struct cudaTextureDesc texDesc;

  memset(&resDesc, 0, sizeof(resDesc));
  memset(&texDesc, 0, sizeof(texDesc));

  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.desc =
      (sizeof_alias == 4
           ? cudaCreateChannelDesc<int>()
           : (sizeof_alias == 8
                  ? cudaCreateChannelDesc< ::int2>()
                  :
                  /* sizeof_alias == 16 */ cudaCreateChannelDesc< ::int4>()));
  resDesc.res.linear.sizeInBytes = alloc_size;
  resDesc.res.linear.devPtr      = alloc_ptr;

  CUDA_SAFE_CALL(
      cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr));

  return tex_obj;
}

//==============================================================================
// <editor-fold desc="SharedAllocationRecord::get_label()"> {{{1

std::string SharedAllocationRecord<Kokkos::CudaSpace, void>::get_label() const {
  SharedAllocationHeader header;

  Kokkos::Impl::DeepCopy<Kokkos::HostSpace, Kokkos::CudaSpace>(
      &header, RecordBase::head(), sizeof(SharedAllocationHeader));

  return std::string(header.m_label);
}

std::string SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::get_label()
    const {
  return std::string(RecordBase::head()->m_label);
}

std::string
SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::get_label() const {
  return std::string(RecordBase::head()->m_label);
}

// </editor-fold> end SharedAllocationRecord::get_label() }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="SharedAllocationRecord allocate()"> {{{1

SharedAllocationRecord<Kokkos::CudaSpace, void>
    *SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate(
        const Kokkos::CudaSpace &arg_space, const std::string &arg_label,
        const size_t arg_alloc_size) {
  return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
}

SharedAllocationRecord<Kokkos::CudaUVMSpace, void>
    *SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::allocate(
        const Kokkos::CudaUVMSpace &arg_space, const std::string &arg_label,
        const size_t arg_alloc_size) {
  return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
}

SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>
    *SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::allocate(
        const Kokkos::CudaHostPinnedSpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size) {
  return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
}

// </editor-fold> end SharedAllocationRecord allocate() }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="SharedAllocationRecord deallocate"> {{{1

void SharedAllocationRecord<Kokkos::CudaSpace, void>::deallocate(
    SharedAllocationRecord<void, void> *arg_rec) {
  delete static_cast<SharedAllocationRecord *>(arg_rec);
}

void SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::deallocate(
    SharedAllocationRecord<void, void> *arg_rec) {
  delete static_cast<SharedAllocationRecord *>(arg_rec);
}

void SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::deallocate(
    SharedAllocationRecord<void, void> *arg_rec) {
  delete static_cast<SharedAllocationRecord *>(arg_rec);
}

// </editor-fold> end SharedAllocationRecord deallocate }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="SharedAllocationRecord destructors"> {{{1

SharedAllocationRecord<Kokkos::CudaSpace, void>::~SharedAllocationRecord() {
  const char *label = nullptr;
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    SharedAllocationHeader header;
    Kokkos::Impl::DeepCopy<Kokkos::CudaSpace, HostSpace>(
        &header, RecordBase::m_alloc_ptr, sizeof(SharedAllocationHeader));
    label = header.label();
  }
  auto alloc_size = SharedAllocationRecord<void, void>::m_alloc_size;
  m_space.deallocate(label, SharedAllocationRecord<void, void>::m_alloc_ptr,
                     alloc_size, (alloc_size - sizeof(SharedAllocationHeader)));
}

SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::~SharedAllocationRecord() {
  const char *label = nullptr;
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    label = RecordBase::m_alloc_ptr->m_label;
  }
  m_space.deallocate(label, SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size,
                     (SharedAllocationRecord<void, void>::m_alloc_size -
                      sizeof(SharedAllocationHeader)));
}

SharedAllocationRecord<Kokkos::CudaHostPinnedSpace,
                       void>::~SharedAllocationRecord() {
  m_space.deallocate(RecordBase::m_alloc_ptr->m_label,
                     SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size,
                     (SharedAllocationRecord<void, void>::m_alloc_size -
                      sizeof(SharedAllocationHeader)));
}

// </editor-fold> end SharedAllocationRecord destructors }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="SharedAllocationRecord constructors"> {{{1

SharedAllocationRecord<Kokkos::CudaSpace, void>::SharedAllocationRecord(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_label,
    const size_t arg_alloc_size,
    const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::CudaSpace, void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
      m_tex_obj(0),
      m_space(arg_space) {

  SharedAllocationHeader header;

  // Fill in the Header information
  header.m_record = static_cast<SharedAllocationRecord<void, void> *>(this);

  strncpy(header.m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;

  // Copy to device memory
  Kokkos::Impl::DeepCopy<CudaSpace, HostSpace>(RecordBase::m_alloc_ptr, &header,
                                               sizeof(SharedAllocationHeader));
}

SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::SharedAllocationRecord(
    const Kokkos::CudaUVMSpace &arg_space, const std::string &arg_label,
    const size_t arg_alloc_size,
    const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
      m_tex_obj(0),
      m_space(arg_space) {
  // Fill in the Header information, directly accessible via UVM

  RecordBase::m_alloc_ptr->m_record = this;

  strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);

  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr
      ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
}

SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::
    SharedAllocationRecord(
        const Kokkos::CudaHostPinnedSpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::CudaHostPinnedSpace,
                                  void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
      m_space(arg_space) {
  // Fill in the Header information, directly accessible on the host

  RecordBase::m_alloc_ptr->m_record = this;

  strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr
      ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
}

// </editor-fold> end SharedAllocationRecord constructors }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="SharedAllocationRecored::(re|de|)allocate_tracked"> {{{1

void *SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return nullptr;

  //printf("running allocate tracker\n");
  SharedAllocationRecord *const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_multi_dev(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::Kokkos_multi_dev_mem* space,
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return;

  //printf("running allocate tracked multi dev\n");

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);
  space->num_dev = gpuNum;
  //printf("Num devices = %d = %d\n", space->num_dev, gpuNum);

  size_t arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;

  omp_set_num_threads(gpuNum); 
  #pragma omp parallel
  {
    cudaSetDevice(omp_get_thread_num());
    cudaMalloc((void**)&space->space[omp_get_thread_num()], arg_alloc_size_multi_dev);
  }
  
  for (int i = gpuNum; i < 12; i++)
    space->space[i] = NULL;
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_smulti_dev(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::sKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return;

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);
  int arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;
 
  omp_set_num_threads(gpuNum);
  #pragma omp parallel
  {
    float* array; 
    cudaSetDevice(omp_get_thread_num());
    cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
    cudaMalloc((void**)&array, arg_alloc_size_multi_dev);
    cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
    //cudaMalloc(&space[omp_get_thread_num()]->array, arg_alloc_size_multi_dev);
  }
}

//template <typename T>
void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_multi_dev_test(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos_multi_dev_mem_test<float>* (&space)[12],
    //T* (&space)[12],
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return;

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);
  int arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;

  omp_set_num_threads(gpuNum);
  #pragma omp parallel
  {
    float* array; 
    cudaSetDevice(omp_get_thread_num());
    cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::Kokkos_multi_dev_mem_test<float>));
    cudaMalloc((void**)&array, arg_alloc_size_multi_dev);
    cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
    //cudaMalloc(&space[omp_get_thread_num()]->array, arg_alloc_size_multi_dev);
  }
}

//template <typename T>
void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_multi_dev_test(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos_multi_dev_mem_test<double>* (&space)[12],
    //T* (&space)[12],
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return;

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);
  int arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;

  omp_set_num_threads(gpuNum);
  #pragma omp parallel
  {
    double* array; 
    cudaSetDevice(omp_get_thread_num());
    cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::Kokkos_multi_dev_mem_test<double>));
    cudaMalloc((void**)&array, arg_alloc_size_multi_dev);
    cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
    //cudaMalloc(&space[omp_get_thread_num()]->array, arg_alloc_size_multi_dev);
  }
}



void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_dmulti_dev(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::dKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return;

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);

  int arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;

  omp_set_num_threads(gpuNum);
  #pragma omp parallel
  {
    double* array; 
    cudaSetDevice(omp_get_thread_num());
    cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::dKokkos_multi_dev_mem));
    cudaMalloc((void**)&array, arg_alloc_size_multi_dev);
    cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_multi_dev_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::Kokkos_multi_dev_mem* space,
    const size_t arg_alloc_size,
    int num_dev) {
  if (!arg_alloc_size) return;

  //printf("running allocate tracked multi dev\n");

  size_t arg_alloc_size_multi_dev = arg_alloc_size / num_dev;

  omp_set_num_threads(num_dev); 
  #pragma omp parallel
  {
    cudaSetDevice(omp_get_thread_num());
    cudaMalloc((void**)&space->space[omp_get_thread_num()], arg_alloc_size_multi_dev);
  }
  
  for (int i = num_dev; i < 12; i++)
    space->space[i] = NULL;
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_smulti_dev_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::sKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size, int num_dev) {
  if (!arg_alloc_size) return;

  int arg_alloc_size_multi_dev = arg_alloc_size / num_dev;

  omp_set_num_threads(num_dev);
  #pragma omp parallel
  {
    float* array; 
    cudaSetDevice(omp_get_thread_num());
    cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
    cudaMalloc((void**)&array, arg_alloc_size_multi_dev);
    cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_dmulti_dev_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::dKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size, int num_dev) {
  if (!arg_alloc_size) return;

  int arg_alloc_size_multi_dev = arg_alloc_size / num_dev;

  omp_set_num_threads(num_dev);
  #pragma omp parallel
  {
    double* array; 
    cudaSetDevice(omp_get_thread_num());
    cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::dKokkos_multi_dev_mem));
    cudaMalloc((void**)&array, arg_alloc_size_multi_dev);
    cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_multi_dev_stencil(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::Kokkos_multi_dev_mem* space,
    const size_t arg_alloc_size,
    int size_of_stencil) {
  if (!arg_alloc_size) return;

  //printf("running allocate tracked multi dev\n");

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);
  space->num_dev = gpuNum;
  //printf("Num devices = %d = %d\n", space->num_dev, gpuNum);

  size_t arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;

  omp_set_num_threads(gpuNum); 
  #pragma omp parallel
  {
    cudaSetDevice(omp_get_thread_num());
    if ( omp_get_thread_num() == 0 )
    {
      cudaMalloc((void**)&space->space[omp_get_thread_num()], arg_alloc_size_multi_dev + size_of_stencil*sizeof(float));
    }
    else if( omp_get_thread_num() == gpuNum - 1 )
    {
      cudaMalloc((void**)&space->space[omp_get_thread_num()], arg_alloc_size_multi_dev + size_of_stencil*sizeof(float));

    }
    else
    {
      cudaMalloc( (void**)&space->space[omp_get_thread_num()], arg_alloc_size_multi_dev + ( 2 * size_of_stencil)*sizeof(float) );
    }
  }
  
  for (int i = gpuNum; i < 12; i++)
    space->space[i] = NULL;
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_smulti_dev_stencil(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::sKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size,
    int size_of_stencil) {
  if (!arg_alloc_size) return;

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);

  size_t arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;

  omp_set_num_threads(gpuNum);
  #pragma omp parallel
  {
    if ( omp_get_thread_num() == 0 )
    {
      float* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(float));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
      //sarray_pointers[omp_get_thread_num()] = array;
    }
    else if( omp_get_thread_num() == gpuNum - 1 )
    {
      float* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(float));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
      //sarray_pointers[omp_get_thread_num()] = array;
    }
    else
    {
      float* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + ( 2 * size_of_stencil ) * sizeof(float));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
      //sarray_pointers[omp_get_thread_num()] = array;
    }
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_dmulti_dev_stencil(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::dKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size,
    int size_of_stencil) {
  if (!arg_alloc_size) return;

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);

  size_t arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;

  omp_set_num_threads(gpuNum);
  #pragma omp parallel
  {
    if ( omp_get_thread_num() == 0 )
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(double));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      //darray_pointers[omp_get_thread_num()] = array;
      //printf("Pointer = %p\n", array);
    }
    else if( omp_get_thread_num() == gpuNum - 1 )
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(double));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      //darray_pointers[omp_get_thread_num()] = array;
    }
    else
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + ( 2 * size_of_stencil ) * sizeof(double));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      //darray_pointers[omp_get_thread_num()] = array;
    }
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_ddmulti_dev_stencil(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::ddKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size,
    int size_of_stencil) {
  if (!arg_alloc_size) return;

  int gpuNum;
  cudaGetDeviceCount(&gpuNum);

  size_t arg_alloc_size_multi_dev = arg_alloc_size / gpuNum;

  omp_set_num_threads(gpuNum);
  #pragma omp parallel
  {
    if ( omp_get_thread_num() == 0 )
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(double));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      cudaMalloc((void**)&space[omp_get_thread_num()+1], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(double));
      cudaMemcpy(&(space[omp_get_thread_num()+1]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      //darray_pointers[omp_get_thread_num()] = array;
      //printf("Pointer = %p\n", array);
      //printf("Pointer = %p\n", &(space[omp_get_thread_num()]->array));
      
      //cudaMemcpyPeer(&(space[0]->array[0]), 0, &(space[1]->array[0]), 0, size_of_stencil*sizeof(double) );
    }
    /*
    else if( omp_get_thread_num() == gpuNum - 1 )
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&space[omp_get_thread_num()]->array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(double));
      //cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      darray_pointers[omp_get_thread_num()] = array;
    }
    else
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&space[omp_get_thread_num()]->array, arg_alloc_size_multi_dev + ( 2 * size_of_stencil ) * sizeof(double));
      //cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      darray_pointers[omp_get_thread_num()] = array;
    }
    */
  }
}



void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_multi_dev_stencil_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::Kokkos_multi_dev_mem* space,
    const size_t arg_alloc_size,
    int size_of_stencil,
    int num_dev) {
  if (!arg_alloc_size) return;

  //printf("running allocate tracked multi dev\n");

  space->num_dev = num_dev;
  //printf("Num devices = %d\n", num_dev);
  //printf("Bytes = %d\n", arg_alloc_size / num_dev);

  if(num_dev == 1)
  {
    cudaMalloc((void**)&space->space[omp_get_thread_num()], arg_alloc_size);
    //printf("Bytes = %d, num float = %d\n", arg_alloc_size, arg_alloc_size / 4);
  }
  else
  {
    size_t arg_alloc_size_multi_dev = arg_alloc_size / num_dev;
    omp_set_num_threads(num_dev); 
    #pragma omp parallel
    {
      cudaSetDevice(omp_get_thread_num());
      if ( omp_get_thread_num() == 0 )
      {
        cudaMalloc((void**)&space->space[omp_get_thread_num()], arg_alloc_size_multi_dev + size_of_stencil*sizeof(float));
        //printf("OpenMP %d -> Bytes = %d\n", omp_get_thread_num(), arg_alloc_size_multi_dev + size_of_stencil*sizeof(float));
      }
      else if( omp_get_thread_num() == num_dev - 1 )
      {
        cudaMalloc((void**)&space->space[omp_get_thread_num()], arg_alloc_size_multi_dev + size_of_stencil *sizeof(float));
        //printf("OpenMP %d -> Bytes = %d\n", omp_get_thread_num(), arg_alloc_size_multi_dev + size_of_stencil*sizeof(float));
      }
      else
      {
        cudaMalloc( (void**)&space->space[omp_get_thread_num()], arg_alloc_size_multi_dev + ( 2 * size_of_stencil)*sizeof(float) );
        //printf("OpenMP %d -> Bytes = %d\n", omp_get_thread_num(),  arg_alloc_size_multi_dev + ( 2 * size_of_stencil)*sizeof(float) );
      }
    }
  
    for (int i = num_dev; i < 12; i++)
      space->space[i] = NULL;
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_smulti_dev_stencil_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::sKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size,
    int size_of_stencil,
    int num_dev) {
  if (!arg_alloc_size) return;

  size_t arg_alloc_size_multi_dev = arg_alloc_size / num_dev;

  omp_set_num_threads(num_dev);
  #pragma omp parallel
  {
    if ( omp_get_thread_num() == 0 )
    {
      float* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(float));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
      //sarray_pointers[omp_get_thread_num()] = array;
    }
    else if( omp_get_thread_num() == num_dev - 1 )
    {
      float* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(float));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
      //sarray_pointers[omp_get_thread_num()] = array;
    }
    else
    {
      float* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + ( 2 * size_of_stencil ) * sizeof(float));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(float*), cudaMemcpyHostToDevice);
      //sarray_pointers[omp_get_thread_num()] = array;
    }
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::allocate_tracked_dmulti_dev_stencil_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::dKokkos_multi_dev_mem* (&space)[12],
    const size_t arg_alloc_size,
    int size_of_stencil,
    int num_dev) {
  if (!arg_alloc_size) return;

  size_t arg_alloc_size_multi_dev = arg_alloc_size / num_dev;

  omp_set_num_threads(num_dev);
  #pragma omp parallel
  {
    if ( omp_get_thread_num() == 0 )
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(double));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      //darray_pointers[omp_get_thread_num()] = array;
    }
    else if( omp_get_thread_num() == num_dev - 1 )
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + size_of_stencil * sizeof(double));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      //darray_pointers[omp_get_thread_num()] = array;
    }
    else
    {
      double* array; 
      cudaSetDevice(omp_get_thread_num());
      cudaMalloc((void**)&space[omp_get_thread_num()], sizeof(Kokkos::sKokkos_multi_dev_mem));
      cudaMalloc((void**)&array, arg_alloc_size_multi_dev + ( 2 * size_of_stencil ) * sizeof(double));
      cudaMemcpy(&(space[omp_get_thread_num()]->array), &array, sizeof(double*), cudaMemcpyHostToDevice);
      //darray_pointers[omp_get_thread_num()] = array;
    }
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::multi_dev_stencil_ghost(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::Kokkos_multi_dev_mem* space,
    int size_array,
    int size_of_stencil) {

  int gpuNum, size_array_dev;
  cudaGetDeviceCount(&gpuNum);
  space->num_dev = gpuNum;
  size_array_dev = size_array / gpuNum;

  //printf("Num devices = %d = %d\n", space->num_dev, gpuNum);
 
  if ( gpuNum > 1 )
  {
    omp_set_num_threads(gpuNum); 
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        cudaMemcpyPeer(                              &space->space[id + 1][0], 
                                                                       id + 1,
                        &space->space[id][(size_array_dev - size_of_stencil)],   
                                                                           id,
                                   size_of_stencil*sizeof(space->space[id][0]) );
      }
      else if( id == gpuNum - 1 )
      {
        if (id - 1 == 0)
        {
          cudaMemcpyPeer(      &space->space[id - 1][size_array_dev], 
                                                              id - 1,
                                  &space->space[id][size_of_stencil],   
                                                                  id,
                          size_of_stencil*sizeof(space->space[id][0]) );
        }
        else
        {
          cudaMemcpyPeer( &space->space[id - 1][size_array_dev+size_of_stencil], 
                                                                         id - 1,
                                             &space->space[id][size_of_stencil],   
                                                                             id,
                                     size_of_stencil*sizeof(space->space[id][0]) );
        }
      } 
      else
      {
        cudaMemcpyPeer(                              &space->space[id + 1][0], 
                                                                       id + 1,
                        &space->space[id][(size_array_dev - size_of_stencil)],   
                                                                           id,
                                   size_of_stencil*sizeof(space->space[id][0]) );

        cudaMemcpyPeer( &space->space[id - 1][size_array_dev+size_of_stencil], 
                                                                       id - 1,
                                           &space->space[id][size_of_stencil],   
                                                                           id,
                                   size_of_stencil*sizeof(space->space[id][0]) );
      }
      cudaDeviceSynchronize(); 
    }
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::multi_dev_stencil_sghost(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::sKokkos_multi_dev_mem* (&space)[12],
    int size_array,
    int size_of_stencil) {

  int gpuNum, size_array_dev;
  cudaGetDeviceCount(&gpuNum);
  size_array_dev = size_array / gpuNum;

  if ( gpuNum > 1 )
  {
    omp_set_num_threads(gpuNum);
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        float *array_pointer_src; 
        float *array_pointer_dst; 
        
        cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst, &(space[id + 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        //sarray_pointers[id];
        //sarray_pointers[id+1];
        //printf("Array pointer src= %p\n", array_pointer_src + (size_array_dev - size_of_stencil));
        //printf("Array pointer src2= %p\n", array_pointer_src + ((size_array_dev - size_of_stencil)*sizeof(float)));
        //printf("offset= %d\n", (size_array_dev - size_of_stencil));
        //printf("offset2= %d\n", (size_array_dev - size_of_stencil)*sizeof(float));
        //printf("Array pointer dst= %p\n", array_pointer_dst);
        
        cudaMemcpyPeer(                                      (array_pointer_dst),
                                                                          id + 1,
                        (array_pointer_src) + (size_array_dev - size_of_stencil),
                                                                              id,
                                                    size_of_stencil*sizeof(float) );
      }
      else if( id == gpuNum - 1 )
      {
        if (id - 1 == 0)
        {
          float *array_pointer_src; 
          float *array_pointer_dst; 
        
          cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(float*), cudaMemcpyDeviceToHost);
          cudaMemcpy(&array_pointer_dst, &(space[id - 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
          
          cudaMemcpyPeer(  (array_pointer_dst)+(size_array_dev),
                                                         id - 1,
                          (array_pointer_src)+(size_of_stencil),
                                                             id,
                                   size_of_stencil*sizeof(float) );
        }
        else
        {
          float *array_pointer_src; 
          float *array_pointer_dst; 
        
          cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(float*), cudaMemcpyDeviceToHost);
          cudaMemcpy(&array_pointer_dst, &(space[id - 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
          
          cudaMemcpyPeer( (array_pointer_dst)+(size_array_dev+size_of_stencil),
                                                                        id - 1,
                                         (array_pointer_src)+(size_of_stencil),
                                                                            id,
                                                  size_of_stencil*sizeof(float) );
        }
      }
      else
      {
        float *array_pointer_src; 
        float *array_pointer_dst_1; 
        float *array_pointer_dst_2; 
        
        cudaMemcpy(  &array_pointer_src,     &(space[id]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst_1, &(space[id - 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst_2, &(space[id + 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        
        cudaMemcpyPeer(                                  (array_pointer_dst_2),
                                                                        id + 1,
                        (array_pointer_src)+(size_array_dev - size_of_stencil),
                                                                            id,
                                                size_of_stencil*sizeof(float) );

        cudaMemcpyPeer( (array_pointer_dst_1)+(size_array_dev+size_of_stencil),
                                                                        id - 1,
                                         (array_pointer_src)+(size_of_stencil),
                                                                            id,
                                                size_of_stencil*sizeof(float) );
      }
      cudaDeviceSynchronize();
    }
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::multi_dev_stencil_dghost(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::dKokkos_multi_dev_mem* (&space)[12],
    int size_array,
    int size_of_stencil) {

  int gpuNum, size_array_dev;
  cudaGetDeviceCount(&gpuNum);
  size_array_dev = size_array / gpuNum;

  if ( gpuNum > 1 )
  {
    omp_set_num_threads(gpuNum);
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        double *array_pointer_src; 
        double *array_pointer_dst; 

        cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst, &(space[id + 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        
        //printf("Array pointer src= %p = %p ??\n", &(space[id]->array), (space[id]->array));
        //printf("Array pointer src= %p = %p ??\n", array_pointer_src, darray_pointers[id]);
        //printf("Array pointer dst= %p = %p ??\n", array_pointer_dst, darray_pointers[id+1]);
        
        cudaMemcpyPeer(                                      (array_pointer_dst),
                                                                          id + 1,
                        (array_pointer_src) + (size_array_dev - size_of_stencil),
                                                                              id,
                                                 size_of_stencil*sizeof(double) );
      }
      else if( id == gpuNum - 1 )
      {
        if (id - 1 == 0)
        {
          double *array_pointer_src; 
          double *array_pointer_dst; 
        
          cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(double*), cudaMemcpyDeviceToHost);
          cudaMemcpy(&array_pointer_dst, &(space[id - 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
          
          cudaMemcpyPeer(  (array_pointer_dst)+(size_array_dev),
                                                         id - 1,
                          (array_pointer_src)+(size_of_stencil),
                                                             id,
                                size_of_stencil*sizeof(double) );
        }
        else
        {
          double *array_pointer_src; 
          double *array_pointer_dst; 
        
          cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(double*), cudaMemcpyDeviceToHost);
          cudaMemcpy(&array_pointer_dst, &(space[id - 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
          
          cudaMemcpyPeer( (array_pointer_dst)+(size_array_dev+size_of_stencil),
                                                                        id - 1,
                                         (array_pointer_src)+(size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );
        }
      }
      else
      {
        double *array_pointer_src; 
        double *array_pointer_dst_1; 
        double *array_pointer_dst_2; 
        
        cudaMemcpy(  &array_pointer_src,     &(space[id]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst_1, &(space[id - 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst_2, &(space[id + 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        
        cudaMemcpyPeer(                                  (array_pointer_dst_2),
                                                                        id + 1,
                        (array_pointer_src)+(size_array_dev - size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );

        cudaMemcpyPeer( (array_pointer_dst_1)+(size_array_dev+size_of_stencil),
                                                                        id - 1,
                                         (array_pointer_src)+(size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );
      }
      cudaDeviceSynchronize();
    }
  }

  /*
  if ( gpuNum > 1 )
  {
    omp_set_num_threads(gpuNum);
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        cudaMemcpyPeer(                                &(space[id + 1]->array),
                                                                        id + 1,
                        &(space[id]->array)+(size_array_dev - size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );
       
      }
      else if( id == gpuNum - 1 )
      {
        if (id - 1 == 0)
        {
          cudaMemcpyPeer( &(space[id - 1]->array)+(size_array_dev),
                                                            id - 1,
                             &(space[id]->array)+(size_of_stencil),
                                                                id,
                                   size_of_stencil*sizeof(double) );
        }
        else
        {
          cudaMemcpyPeer(  &(space[id - 1]->array)+(size_array_dev+size_of_stencil),
                                                                             id - 1,
                                              &(space[id]->array)+(size_of_stencil),
                                                                                 id,
                                                    size_of_stencil*sizeof(double) );
        }
      }
      else
      {
        cudaMemcpyPeer(                                &(space[id + 1]->array),
                                                                        id + 1,
                        &(space[id]->array)+(size_array_dev - size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );

        cudaMemcpyPeer( &(space[id - 1]->array)+(size_array_dev+size_of_stencil),
                                                                          id - 1,
                                           &(space[id]->array)+(size_of_stencil),
                                                                              id,
                                                 size_of_stencil*sizeof(double) );
      }
      cudaDeviceSynchronize();
    }
  }
  */
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::multi_dev_stencil_sghost_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::sKokkos_multi_dev_mem* (&space)[12],
    int size_array,
    int size_of_stencil,
    int num_dev) {

  int size_array_dev;
  size_array_dev = size_array / num_dev;

  if ( num_dev > 1 )
  {
    omp_set_num_threads(num_dev);
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        float *array_pointer_src; 
        float *array_pointer_dst; 
        
        cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst, &(space[id + 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        
        cudaMemcpyPeer(                                      (array_pointer_dst),
                                                                          id + 1,
                        (array_pointer_src) + (size_array_dev - size_of_stencil),
                                                                              id,
                                                    size_of_stencil*sizeof(float) );
      }
      else if( id == num_dev - 1 )
      {
        if (id - 1 == 0)
        {
          float *array_pointer_src; 
          float *array_pointer_dst; 
        
          cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(float*), cudaMemcpyDeviceToHost);
          cudaMemcpy(&array_pointer_dst, &(space[id - 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
          
          cudaMemcpyPeer(  (array_pointer_dst)+(size_array_dev),
                                                         id - 1,
                          (array_pointer_src)+(size_of_stencil),
                                                             id,
                                   size_of_stencil*sizeof(float) );
        }
        else
        {
          float *array_pointer_src; 
          float *array_pointer_dst; 
        
          cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(float*), cudaMemcpyDeviceToHost);
          cudaMemcpy(&array_pointer_dst, &(space[id - 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
          
          cudaMemcpyPeer( (array_pointer_dst)+(size_array_dev+size_of_stencil),
                                                                        id - 1,
                                         (array_pointer_src)+(size_of_stencil),
                                                                            id,
                                                  size_of_stencil*sizeof(float) );
        }
      }
      else
      {
        float *array_pointer_src; 
        float *array_pointer_dst_1; 
        float *array_pointer_dst_2; 
        
        cudaMemcpy(  &array_pointer_src,     &(space[id]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst_1, &(space[id - 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst_2, &(space[id + 1]->array), sizeof(float*), cudaMemcpyDeviceToHost);
        
        cudaMemcpyPeer(                                  (array_pointer_dst_2),
                                                                        id + 1,
                        (array_pointer_src)+(size_array_dev - size_of_stencil),
                                                                            id,
                                                size_of_stencil*sizeof(float) );

        cudaMemcpyPeer( (array_pointer_dst_1)+(size_array_dev+size_of_stencil),
                                                                        id - 1,
                                         (array_pointer_src)+(size_of_stencil),
                                                                            id,
                                                size_of_stencil*sizeof(float) );
      }
      cudaDeviceSynchronize();
    }
  }

  /*
  if ( num_dev > 1 )
  {
    omp_set_num_threads(num_dev);
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        cudaMemcpyPeer(                                &(space[id + 1]->array),
                                                                        id + 1,
                        &(space[id]->array)+(size_array_dev - size_of_stencil),
                                                                            id,
                                                size_of_stencil*sizeof(float) );
       
      }
      else if( id == num_dev - 1 )
      {
        if (id - 1 == 0)
        {
          cudaMemcpyPeer( &(space[id - 1]->array)+(size_array_dev),
                                                            id - 1,
                             &(space[id]->array)+(size_of_stencil),
                                                                id,
                                    size_of_stencil*sizeof(float) );
        }
        else
        {
          cudaMemcpyPeer( &(space[id - 1]->array)+(size_array_dev+size_of_stencil),
                                                                            id - 1,
                                             &(space[id]->array)+(size_of_stencil),
                                                                                id,
                                                    size_of_stencil*sizeof(float) );
        }
      }
      else
      {
        cudaMemcpyPeer(                                &(space[id + 1]->array),
                                                                        id + 1,
                        &(space[id]->array)+(size_array_dev - size_of_stencil),
                                                                            id,
                                                size_of_stencil*sizeof(float) );

        cudaMemcpyPeer( &(space[id - 1]->array)+(size_array_dev+size_of_stencil),
                                                                          id - 1,
                                           &(space[id]->array)+(size_of_stencil),
                                                                              id,
                                                  size_of_stencil*sizeof(float) );
      }
      cudaDeviceSynchronize();
    }
  }
  */
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::multi_dev_stencil_dghost_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::dKokkos_multi_dev_mem* (&space)[12],
    int size_array,
    int size_of_stencil,
    int num_dev) {

  int size_array_dev;
  size_array_dev = size_array / num_dev;

  if ( num_dev > 1 )
  {
    omp_set_num_threads(num_dev);
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        double *array_pointer_src; 
        double *array_pointer_dst; 
        
        cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst, &(space[id + 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        
        cudaMemcpyPeer(                                      (array_pointer_dst),
                                                                          id + 1,
                        (array_pointer_src) + (size_array_dev - size_of_stencil),
                                                                              id,
                                                 size_of_stencil*sizeof(double) );
      }
      else if( id == num_dev - 1 )
      {
        if (id - 1 == 0)
        {
          double *array_pointer_src; 
          double *array_pointer_dst; 
        
          cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(double*), cudaMemcpyDeviceToHost);
          cudaMemcpy(&array_pointer_dst, &(space[id - 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
          
          cudaMemcpyPeer(  (array_pointer_dst)+(size_array_dev),
                                                         id - 1,
                          (array_pointer_src)+(size_of_stencil),
                                                             id,
                                size_of_stencil*sizeof(double) );
        }
        else
        {
          double *array_pointer_src; 
          double *array_pointer_dst; 
        
          cudaMemcpy(&array_pointer_src,     &(space[id]->array), sizeof(double*), cudaMemcpyDeviceToHost);
          cudaMemcpy(&array_pointer_dst, &(space[id - 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
          
          cudaMemcpyPeer( (array_pointer_dst)+(size_array_dev+size_of_stencil),
                                                                        id - 1,
                                         (array_pointer_src)+(size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );
        }
      }
      else
      {
        double *array_pointer_src; 
        double *array_pointer_dst_1; 
        double *array_pointer_dst_2; 
        
        cudaMemcpy(  &array_pointer_src,     &(space[id]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst_1, &(space[id - 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&array_pointer_dst_2, &(space[id + 1]->array), sizeof(double*), cudaMemcpyDeviceToHost);
        
        cudaMemcpyPeer(                                  (array_pointer_dst_2),
                                                                        id + 1,
                        (array_pointer_src)+(size_array_dev - size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );

        cudaMemcpyPeer( (array_pointer_dst_1)+(size_array_dev+size_of_stencil),
                                                                        id - 1,
                                         (array_pointer_src)+(size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );
      }
      cudaDeviceSynchronize();
    }
  }

  /*
  if ( num_dev > 1 )
  {
    omp_set_num_threads(num_dev);
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        cudaMemcpyPeer(                                &(space[id + 1]->array),
                                                                        id + 1,
                        &(space[id]->array)+(size_array_dev - size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );
       
      }
      else if( id == num_dev - 1 )
      {
        if (id - 1 == 0)
        {
          cudaMemcpyPeer( &(space[id - 1]->array)+(size_array_dev),
                                                            id - 1,
                             &(space[id]->array)+(size_of_stencil),
                                                                id,
                                   size_of_stencil*sizeof(double) );
        }
        else
        {
          cudaMemcpyPeer(  &(space[id - 1]->array)+(size_array_dev+size_of_stencil),
                                                                             id - 1,
                                              &(space[id]->array)+(size_of_stencil),
                                                                                 id,
                                                    size_of_stencil*sizeof(double) );
        }
      }
      else
      {
        cudaMemcpyPeer(                                &(space[id + 1]->array),
                                                                        id + 1,
                        &(space[id]->array)+(size_array_dev - size_of_stencil),
                                                                            id,
                                               size_of_stencil*sizeof(double) );

        cudaMemcpyPeer( &(space[id - 1]->array)+(size_array_dev+size_of_stencil),
                                                                          id - 1,
                                           &(space[id]->array)+(size_of_stencil),
                                                                              id,
                                                 size_of_stencil*sizeof(double) );
      }
      cudaDeviceSynchronize();
    }
  }
  */
}

int SharedAllocationRecord<Kokkos::CudaSpace, void>::multi_dev_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return 0;

  //printf("running allocate tracked multi dev\n");

  int gpuNum;
  int num_dev;
  int threshold = 10485760; //Total L1 cache in V100 (128KB*80SMs)
  cudaGetDeviceCount(&gpuNum);
  num_dev = gpuNum;

  if (arg_alloc_size/num_dev >= threshold)
  {
    num_dev = gpuNum;
  }
  else
  {
    while ( num_dev > 1)
    {
      num_dev--;
      if ( arg_alloc_size/num_dev >= threshold )
        break;
    }
  }

  //printf("Num devices = %d, Num devices used %d\n", gpuNum, num_dev);

  return num_dev;

}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::multi_dev_stencil_ghost_auto(
    const Kokkos::CudaSpace &arg_space, const std::string &arg_alloc_label,
    Kokkos::Kokkos_multi_dev_mem* space,
    int size_array,
    int size_of_stencil,
    int num_dev) {

  int size_array_dev;
  space->num_dev = num_dev;
  size_array_dev = size_array / num_dev;

  //printf("Num devices = %d = %d\n", space->num_dev, gpuNum);

  if ( num_dev > 1 )
  {
    omp_set_num_threads(num_dev); 
    #pragma omp parallel
    //for ( int i = 0; i < gpuNum; i++ )
    {
      int id = omp_get_thread_num();
      //int id = i;
      //printf("OpenMP id = %d\n", id);
      cudaSetDevice(omp_get_thread_num());
      if ( id == 0 )
      {
        cudaMemcpyPeer(                              &space->space[id + 1][0], 
                                                                       id + 1,
                        &space->space[id][(size_array_dev - size_of_stencil)],   
                                                                           id,
                                   size_of_stencil*sizeof(space->space[id][0]) );
                          //size_of_stencil*sizeof(float) );
      }
      else if( id == num_dev - 1 )
      {
        if (id - 1 == 0)
        {
          //printf("I am %d sending %1.2f to %d\n", id, space->space[id][size_of_stencil], id-1);
          cudaMemcpyPeer(      &space->space[id - 1][size_array_dev], 
          //cudaMemcpyPeer(      &space->space[id - 1][32], 
                                                              id - 1,
                                  &space->space[id][size_of_stencil],   
                                                                  id,
                          size_of_stencil*sizeof(space->space[id][0]) );
                          //size_of_stencil*sizeof(float) );
        }
        else
        {
          cudaMemcpyPeer( &space->space[id - 1][size_array_dev+size_of_stencil], 
                                                                         id - 1,
                                             &space->space[id][size_of_stencil],   
                                                                             id,
                                     size_of_stencil*sizeof(space->space[id][0]) );
        }
      } 
      else
      {
        cudaMemcpyPeer(                              &space->space[id + 1][0], 
                                                                       id + 1,
                        &space->space[id][(size_array_dev - size_of_stencil)],   
                                                                           id,
                                   size_of_stencil*sizeof(space->space[id][0]) );

        cudaMemcpyPeer( &space->space[id - 1][size_array_dev+size_of_stencil], 
                                                                       id - 1,
                                           &space->space[id][size_of_stencil],   
                                                                           id,
                                   size_of_stencil*sizeof(space->space[id][0]) );
      }
      cudaDeviceSynchronize(); 
    }
  }
}

void SharedAllocationRecord<Kokkos::CudaSpace, void>::deallocate_tracked(
    void *const arg_alloc_ptr) {
  if (arg_alloc_ptr != nullptr) {
    SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void *SharedAllocationRecord<Kokkos::CudaSpace, void>::reallocate_tracked(
    void *const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord *const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<CudaSpace, CudaSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

void *SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::allocate_tracked(
    const Kokkos::CudaUVMSpace &arg_space, const std::string &arg_alloc_label,
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return nullptr;

  SharedAllocationRecord *const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::deallocate_tracked(
    void *const arg_alloc_ptr) {
  if (arg_alloc_ptr != nullptr) {
    SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void *SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::reallocate_tracked(
    void *const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord *const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<CudaUVMSpace, CudaUVMSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

void *
SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::allocate_tracked(
    const Kokkos::CudaHostPinnedSpace &arg_space,
    const std::string &arg_alloc_label, const size_t arg_alloc_size) {
  if (!arg_alloc_size) return nullptr;

  SharedAllocationRecord *const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<Kokkos::CudaHostPinnedSpace,
                            void>::deallocate_tracked(void *const
                                                          arg_alloc_ptr) {
  if (arg_alloc_ptr != nullptr) {
    SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void *
SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::reallocate_tracked(
    void *const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord *const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<CudaHostPinnedSpace, CudaHostPinnedSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

// </editor-fold> end SharedAllocationRecored::(re|de|)allocate_tracked }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="SharedAllocationRecord::get_record()"> {{{1

SharedAllocationRecord<Kokkos::CudaSpace, void> *
SharedAllocationRecord<Kokkos::CudaSpace, void>::get_record(void *alloc_ptr) {
  using RecordCuda = SharedAllocationRecord<Kokkos::CudaSpace, void>;

  using Header = SharedAllocationHeader;

  // Copy the header from the allocation
  Header head;

  Header const *const head_cuda =
      alloc_ptr ? Header::get_header(alloc_ptr) : nullptr;

  if (alloc_ptr) {
    Kokkos::Impl::DeepCopy<HostSpace, CudaSpace>(
        &head, head_cuda, sizeof(SharedAllocationHeader));
  }

  RecordCuda *const record =
      alloc_ptr ? static_cast<RecordCuda *>(head.m_record) : nullptr;

  if (!alloc_ptr || record->m_alloc_ptr != head_cuda) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::CudaSpace , "
                    "void >::get_record ERROR"));
  }

  return record;
}

SharedAllocationRecord<Kokkos::CudaUVMSpace, void> *SharedAllocationRecord<
    Kokkos::CudaUVMSpace, void>::get_record(void *alloc_ptr) {
  using Header     = SharedAllocationHeader;
  using RecordCuda = SharedAllocationRecord<Kokkos::CudaUVMSpace, void>;

  Header *const h =
      alloc_ptr ? reinterpret_cast<Header *>(alloc_ptr) - 1 : nullptr;

  if (!alloc_ptr || h->m_record->m_alloc_ptr != h) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord< "
                    "Kokkos::CudaUVMSpace , void >::get_record ERROR"));
  }

  return static_cast<RecordCuda *>(h->m_record);
}

SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>
    *SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::get_record(
        void *alloc_ptr) {
  using Header     = SharedAllocationHeader;
  using RecordCuda = SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>;

  Header *const h =
      alloc_ptr ? reinterpret_cast<Header *>(alloc_ptr) - 1 : nullptr;

  if (!alloc_ptr || h->m_record->m_alloc_ptr != h) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord< "
                    "Kokkos::CudaHostPinnedSpace , void >::get_record ERROR"));
  }

  return static_cast<RecordCuda *>(h->m_record);
}

// </editor-fold> end SharedAllocationRecord::get_record() }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="SharedAllocationRecord::print_records()"> {{{1

// Iterate records to print orphaned memory ...
void SharedAllocationRecord<Kokkos::CudaSpace, void>::print_records(
    std::ostream &s, const Kokkos::CudaSpace &, bool detail) {
  (void)s;
  (void)detail;
#ifdef KOKKOS_ENABLE_DEBUG
  SharedAllocationRecord<void, void> *r = &s_root_record;

  char buffer[256];

  SharedAllocationHeader head;

  if (detail) {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<HostSpace, CudaSpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));
      } else {
        head.m_label[0] = 0;
      }

      // Formatting dependent on sizeof(uintptr_t)
      const char *format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string =
            "Cuda addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx "
            "+ %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string =
            "Cuda addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ "
            "0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf(buffer, 256, format_string, reinterpret_cast<uintptr_t>(r),
               reinterpret_cast<uintptr_t>(r->m_prev),
               reinterpret_cast<uintptr_t>(r->m_next),
               reinterpret_cast<uintptr_t>(r->m_alloc_ptr), r->m_alloc_size,
               r->m_count, reinterpret_cast<uintptr_t>(r->m_dealloc),
               head.m_label);
      s << buffer;
      r = r->m_next;
    } while (r != &s_root_record);
  } else {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<HostSpace, CudaSpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));

        // Formatting dependent on sizeof(uintptr_t)
        const char *format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "Cuda [ 0x%.12lx + %ld ] %s\n";
        } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "Cuda [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf(buffer, 256, format_string,
                 reinterpret_cast<uintptr_t>(r->data()), r->size(),
                 head.m_label);
      } else {
        snprintf(buffer, 256, "Cuda [ 0 + 0 ]\n");
      }
      s << buffer;
      r = r->m_next;
    } while (r != &s_root_record);
  }
#else
  Kokkos::Impl::throw_runtime_exception(
      "SharedAllocationHeader<CudaSpace>::print_records only works with "
      "KOKKOS_ENABLE_DEBUG enabled");
#endif
}

void SharedAllocationRecord<Kokkos::CudaUVMSpace, void>::print_records(
    std::ostream &s, const Kokkos::CudaUVMSpace &, bool detail) {
  (void)s;
  (void)detail;
#ifdef KOKKOS_ENABLE_DEBUG
  SharedAllocationRecord<void, void>::print_host_accessible_records(
      s, "CudaUVM", &s_root_record, detail);
#else
  Kokkos::Impl::throw_runtime_exception(
      "SharedAllocationHeader<CudaSpace>::print_records only works with "
      "KOKKOS_ENABLE_DEBUG enabled");
#endif
}

void SharedAllocationRecord<Kokkos::CudaHostPinnedSpace, void>::print_records(
    std::ostream &s, const Kokkos::CudaHostPinnedSpace &, bool detail) {
  (void)s;
  (void)detail;
#ifdef KOKKOS_ENABLE_DEBUG
  SharedAllocationRecord<void, void>::print_host_accessible_records(
      s, "CudaHostPinned", &s_root_record, detail);
#else
  Kokkos::Impl::throw_runtime_exception(
      "SharedAllocationHeader<CudaSpace>::print_records only works with "
      "KOKKOS_ENABLE_DEBUG enabled");
#endif
}

// </editor-fold> end SharedAllocationRecord::print_records() }}}1
//==============================================================================

void cuda_prefetch_pointer(const Cuda &space, const void *ptr, size_t bytes,
                           bool to_device) {
  if ((ptr == nullptr) || (bytes == 0)) return;
  cudaPointerAttributes attr;
  CUDA_SAFE_CALL(cudaPointerGetAttributes(&attr, ptr));
  // I measured this and it turns out prefetching towards the host slows
  // DualView syncs down. Probably because the latency is not too bad in the
  // first place for the pull down. If we want to change that provde
  // cudaCpuDeviceId as the device if to_device is false
#if CUDA_VERSION < 10000
  bool is_managed = attr.isManaged;
#else
  bool is_managed = attr.type == cudaMemoryTypeManaged;
#endif
  if (to_device && is_managed &&
      space.cuda_device_prop().concurrentManagedAccess) {
    CUDA_SAFE_CALL(cudaMemPrefetchAsync(ptr, bytes, space.cuda_device(),
                                        space.cuda_stream()));
  }
}

}  // namespace Impl
}  // namespace Kokkos
#else
void KOKKOS_CORE_SRC_CUDA_CUDASPACE_PREVENT_LINK_ERROR() {}
#endif  // KOKKOS_ENABLE_CUDA
