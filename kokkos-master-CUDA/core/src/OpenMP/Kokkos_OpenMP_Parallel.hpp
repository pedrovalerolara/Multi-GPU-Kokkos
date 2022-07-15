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

#include <sys/time.h>

#ifndef KOKKOS_OPENMP_PARALLEL_HPP
#define KOKKOS_OPENMP_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

#include <omp.h>
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::OpenMP> {
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {
#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
printf("ivdep 0\n");
#pragma ivdep
#endif
#endif
//printf("running for loop 0 from ibeg = %d to iend = %d\n", ibeg, iend);
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {
    const TagType t{};
#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
printf("ivdep 1\n");
#pragma ivdep
#endif
#endif
printf("running for loop 1 from ibeg = %d to iend = %d\n", ibeg, iend);
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork);
    }
  }

 public:
  inline void execute() const {
    enum {
      is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value
    };

    if (OpenMP::in_parallel()) {
      //printf("Hi 1\n");
      exec_range<WorkTag>(m_functor, m_policy.begin(), m_policy.end());
    } else {
      //printf("Hi 2\n");
      OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_for");

      //printf("OpenMP thread pool size = %d\n", OpenMP::impl_thread_pool_size());
#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
      {
        HostThreadTeamData& data = *(m_instance->get_thread_data());

        data.set_work_partition(m_policy.end() - m_policy.begin(),
                                m_policy.chunk_size());

        if (is_dynamic) {
          // Make sure work partition is set before stealing
          if (data.pool_rendezvous()) data.pool_rendezvous_release();
        }

        std::pair<int64_t, int64_t> range(0, 0);

        do {
          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelFor::template exec_range<WorkTag>(
              m_functor, range.first + m_policy.begin(),
              range.second + m_policy.begin());

        } while (is_dynamic && 0 <= range.first);
      }
    }
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {}
};

template <class FunctorType, class... Traits>
class HetParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::OpenMP> {
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {
printf("Hello from Parallel from het 1\n");
printf("Using omp taget parallel for\n");
printf("Last iter = %d\n", iend);
#pragma omp parallel num_threads(2)
{
    printf("Hey I am the OpenMP thread = %d\n", omp_get_thread_num());
    int range = iend / 2;    

    if (omp_get_thread_num() == 0)
    {
       #pragma omp target parallel for map(to: functor) 
       //for (Member iwork = ibeg; iwork < iend; ++iwork) {
       for (Member iwork = ibeg; iwork < range; ++iwork) {
          functor(iwork);
       }
    }
    if (omp_get_thread_num() == 1)
    {
       #pragma omp parallel for  
       for (Member iwork = range; iwork < iend; ++iwork) {
         functor(iwork);
       }
    }
  }
}

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {
      printf("Hello from Parallel from OpenMP 2\n");
    const TagType t{};
#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#endif
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork);
    }
  }

 public:
  inline void execute() const {

    HetParallelFor::template exec_range<WorkTag>(
        m_functor, m_policy.begin(),
        m_policy.end());

  }

  inline HetParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {}
};

template <class FunctorType, class... Traits>
class HetParallelForIndex<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::OpenMP> {
 
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;
  //using Index     = Kokkos::Impl::Index;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
 
  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(int ind, const FunctorType& functor, const Member ibeg,
                 const Member iend) 
  {
    printf("Hello from Parallel from het 1\n");
    printf("Using omp taget parallel for\n");
    printf("Last iter = %d\n", iend);
    printf("Indexx = %d\n", ind);

    #pragma omp parallel num_threads(2)
    {
      printf("Hey I am the OpenMP thread = %d running\n", omp_get_thread_num());
      int range = iend / 2;    

      if (omp_get_thread_num() == 0)
      {
        #pragma omp target parallel for map(to: functor) 
         //for (Member iwork = ibeg; iwork < iend; ++iwork) {
         for (Member iwork = ibeg; iwork < ind; ++iwork) {
            functor(iwork);
         }
      }
      if (omp_get_thread_num() == 1)
      {
        #pragma omp parallel for  
        for (Member iwork = ind; iwork < iend; ++iwork) {
          functor(iwork);
        }
      }
    }
}

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {
      printf("Hello from Parallel from OpenMP 2\n");
    const TagType t{};
#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#endif
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork);
    }
  }

 public:
  
  int ind;

  inline void execute(int ind) const {

    HetParallelForIndex::template exec_range<WorkTag>(
        ind, m_functor, m_policy.begin(),
        m_policy.end());

  }

  inline HetParallelForIndex(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {}
};

#define GIGA 1000000000.0

template <class FunctorType, class... Traits>
class HetParallelForAuto<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::OpenMP> {
 
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;


  inline static int scheduler(int mem, int num_ops, int num_it)
  {
    int ind;

    //Oswald 
    //       -> Intel E5-2683 v4 537.6 GFLOPS or 1,075.2 GFLOPS 
    //       -> NVIDIA P100 4700 GFLOPS
    //       -> PCIe Gen 3.0 32GB/s 
    double CPU_GPU_bandwidth = 32.0;
    double CPU_GFLOPS = 1075.2;
    double GPU_GFLOPS = 4700.0;


    double CPU_side_b = .0;
    double GPU_side_a = .0;   
    double GPU_side_b = .0;   
    double GPU_side_transfer_a = .0;   
    double GPU_side_transfer_b = .0;   

    CPU_side_b = num_ops / ( GIGA * CPU_GFLOPS );
    GPU_side_a = ( num_it * num_ops ) / ( GIGA * GPU_GFLOPS );
    GPU_side_b = num_ops / ( GIGA * GPU_GFLOPS );
    GPU_side_transfer_a = ( num_it * mem ) / ( GIGA * CPU_GPU_bandwidth ); 
    GPU_side_transfer_b =  mem  / ( GIGA * CPU_GPU_bandwidth );

    ind = (GPU_side_a + GPU_side_transfer_a) / (CPU_side_b + GPU_side_b + GPU_side_transfer_b); 

    return(ind);
  }
 
  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(int ind, const FunctorType& functor, const Member ibeg,
                 const Member iend) 
  {
    struct timeval start_cpu, start_gpu, end_cpu, end_gpu, start, end;
    
    int chunk = ind / OpenMP::impl_thread_pool_size();
    printf ( "chunk = %d\n", chunk );

    gettimeofday( &start, NULL );   
    #pragma omp parallel
    #pragma omp master
    {
    //#pragma omp target parallel for map(to: functor) //depend(inout: sentinel) //nowait 
    gettimeofday( &start_gpu, NULL );   
    #pragma omp target teams distribute parallel for map(to: functor) //nowait 
    //#pragma omp target parallel for map(to: functor) depend(inout: sentinel) 
    //for (Member iwork = ind; iwork < iend; ++iwork) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork);
    }
    gettimeofday( &end_gpu, NULL );
    double time_gpu = ( double ) (((end_gpu.tv_sec * 1e6 + end_gpu.tv_usec)
                             - (start_gpu.tv_sec * 1e6 + start_gpu.tv_usec)) / 1e6);
    printf( "time gpu( %e s )\n", time_gpu );
   

    //gettimeofday( &start_cpu, NULL );
    //for ( int i = 0; i < OpenMP::impl_thread_pool_size(); i++ )
    for ( int i = 0; i < 32; i++ )
    {
      #pragma omp task
      {
        for (Member iwork = chunk * i; iwork < chunk * (i + 1); ++iwork) {
          functor(iwork);
        }
      }
    }
 
    /*
    #pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
    {
      int omp_id;
      omp_id = omp_get_thread_num();
      //printf ( "OpenMP Thread = %d\n computing init = %d - end = %d iterations\n", omp_id, chunk * omp_id, chunk * ( omp_id + 1 ) );
      for (Member iwork = chunk * omp_id; iwork < chunk * (omp_id + 1); ++iwork) {
        functor(iwork);
      }

    }
    */ 
    /*
    #pragma omp parallel for //num_threads(32)
    for (Member iwork = ibeg; iwork < ind; ++iwork) {
      //printf("I am the OpenMP thread = %d computing iwork = %d\n", omp_get_thread_num(), iwork);   
      functor(iwork);
    }
    */
    //gettimeofday( &end_cpu, NULL );
    //double time_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
    //                        - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
    //printf( "time cpu( %e s )\n", time_cpu );

    /*
    #pragma omp target depend(in: sentinel)
    {
    if (false)
       printf("hi");
    }
    */
    }
    //#pragma omp barrier
    gettimeofday( &end, NULL );
    double time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                            - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);
    printf( "time( %e s )\n", time );
  }

 public:
  

  inline void execute(int mem, int num_ops) const {

    int ind;
    
    ind = HetParallelForAuto::scheduler(mem, num_ops, m_policy.end() - m_policy.begin());

    printf ("IND = %d\n", ind);

    HetParallelForAuto::template exec_range<WorkTag>(
        ind, m_functor, m_policy.begin(),
        m_policy.end());

  }

  inline HetParallelForAuto(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {}
};


template <class FunctorType, class... Traits>
class HetParallelForIterAuto<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::OpenMP> {
 
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  inline static int scheduler(int mem, int num_ops, int num_it)
  {
    int ind;

    //Oswald 
    //       -> Intel E5-2683 v4 537.6 GFLOPS or 1,075.2 GFLOPS 
    //       -> NVIDIA P100 4700 GFLOPS
    //       -> PCIe Gen 3.0 32GB/s 
    double CPU_GFLOPS = 537.6;
    double GPU_GFLOPS = 4700.0;

    double CPU_side_b = .0;
    double GPU_side_a = .0;   
    double GPU_side_b = .0;   
    double GPU_side_transfer_a = .0;   
    double GPU_side_transfer_b = .0;   

    CPU_side_b = num_ops / ( GIGA * CPU_GFLOPS );
    GPU_side_a = ( num_it * num_ops ) / ( GIGA * GPU_GFLOPS );
    GPU_side_b = num_ops / ( GIGA * GPU_GFLOPS );

    ind = GPU_side_a / ( CPU_side_b + GPU_side_b ); 

    return(ind);
  }
 
  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(int init_iter, int end_iter, int ind, const FunctorType& functor, const Member ibeg,
                 const Member iend) 
  {
    struct timeval start, start_cpu, start_gpu, end, end_cpu, end_gpu;
 
    /*
    FunctorType functor_gpu(functor);
    
    #pragma omp declare target
    functor_gpu;
    #pragma omp end declare target  
    */
    
    /*
    // CPU only 
    for (int iter = init_iter; iter < end_iter; iter++)
    {
      gettimeofday( &start_cpu, NULL );   
      #pragma omp parallel for  
      for (Member iwork = ibeg; iwork < iend; ++iwork) {
        functor(iwork);
      }
      gettimeofday( &end_cpu, NULL );
      double time_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
                              - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
      printf( "time cpu( %e s )\n", time_cpu );
    } 
    */

    //GPU only
    gettimeofday( &start_gpu, NULL );   
    #pragma omp target enter data map(to: functor)
    { 
    for (int iter = init_iter; iter < end_iter; iter++)
    {
      //gettimeofday( &start_gpu, NULL );   
      #pragma omp teams distribute parallel for  
      for (Member iwork = ibeg; iwork < iend; ++iwork) {
        functor(iwork);
      } 
      /*
      gettimeofday( &end_gpu, NULL );
      double time_gpu = ( double ) (((end_gpu.tv_sec * 1e6 + end_gpu.tv_usec)
                              - (start_gpu.tv_sec * 1e6 + start_gpu.tv_usec)) / 1e6);
      printf( "time gpu( %e s )\n", time_gpu );
      */
    }
    }
    #pragma omp target exit data map(from: functor)
    gettimeofday( &end_gpu, NULL );
    double time_gpu = ( double ) (((end_gpu.tv_sec * 1e6 + end_gpu.tv_usec)
                            - (start_gpu.tv_sec * 1e6 + start_gpu.tv_usec)) / 1e6);
    printf( "time gpu( %e s )\n", time_gpu );
   
    /* 
    printf( "cpu + gpu\n" );
    gettimeofday( &start, NULL );   
    //CPU + GPU 
    #pragma omp target enter data map(to: functor)
    { 
    for (int iter = init_iter; iter < end_iter; iter++)
    {
      gettimeofday( &start_gpu, NULL );   
      #pragma omp teams distribute parallel for  
      for (Member iwork = ind; iwork < iend; ++iwork) {
        functor(iwork);
      } 
      gettimeofday( &end_gpu, NULL );
      double time_gpu = ( double ) (((end_gpu.tv_sec * 1e6 + end_gpu.tv_usec)
                              - (start_gpu.tv_sec * 1e6 + start_gpu.tv_usec)) / 1e6);
      printf( "time gpu( %e s )\n", time_gpu );
    }
    }
    #pragma omp target exit data map(from: functor)
    
    // CPU only 
    for (int iter = init_iter; iter < end_iter; iter++)
    {
      gettimeofday( &start_cpu, NULL );   
      #pragma omp parallel for  
      for (Member iwork = ibeg; iwork < ind; ++iwork) {
        functor(iwork);
      }
      gettimeofday( &end_cpu, NULL );
      double time_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
                              - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
      printf( "time cpu( %e s )\n", time_cpu );
    } 

    gettimeofday( &end, NULL );
    double time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                            - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);
    printf( "time( %e s )\n", time );
    */

  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {
    const TagType t{};
#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#endif
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork);
    }
  }

 public:
  

  inline void execute(int init_iter, int end_iter, int mem, int num_ops) const {

    int ind;
    
    ind = HetParallelForIterAuto::scheduler(mem, num_ops, m_policy.end() - m_policy.begin());

    printf("ind = %d\n", ind);

    HetParallelForIterAuto::template exec_range<WorkTag>(
        init_iter, end_iter, ind, m_functor, m_policy.begin(),
        m_policy.end());

  }

  inline HetParallelForIterAuto(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {}
};


// MDRangePolicy impl
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::OpenMP> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using WorkTag       = typename MDRangePolicy::work_tag;

  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
                          // ).set_chunk_size(1) in ctor

  inline static void exec_range(const MDRangePolicy& mdr_policy,
                                const FunctorType& functor, const Member ibeg,
                                const Member iend) {
#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#endif
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      iterate_type(mdr_policy, functor)(iwork);
    }
  }

 public:
  inline void execute() const {
    enum {
      is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value
    };

    if (OpenMP::in_parallel()) {
      ParallelFor::exec_range(m_mdr_policy, m_functor, m_policy.begin(),
                              m_policy.end());
    } else {
      OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_for");

#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
      {
        HostThreadTeamData& data = *(m_instance->get_thread_data());

        data.set_work_partition(m_policy.end() - m_policy.begin(),
                                m_policy.chunk_size());

        if (is_dynamic) {
          // Make sure work partition is set before stealing
          if (data.pool_rendezvous()) data.pool_rendezvous_release();
        }

        std::pair<int64_t, int64_t> range(0, 0);

        do {
          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelFor::exec_range(m_mdr_policy, m_functor,
                                  range.first + m_policy.begin(),
                                  range.second + m_policy.begin());

        } while (is_dynamic && 0 <= range.first);
      }
      // END #pragma omp parallel
    }
  }

  inline ParallelFor(const FunctorType& arg_functor, MDRangePolicy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::OpenMP> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  // Static Assert WorkTag void if ReducerType not InvalidType

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork, update);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update) {
    const TagType t{};
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork, update);
    }
  }

 public:
  inline void execute() const {
    if (m_policy.end() <= m_policy.begin()) {
      if (m_result_ptr) {
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                        m_result_ptr);
        Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
            ReducerConditional::select(m_functor, m_reducer), m_result_ptr);
      }
      return;
    }
    enum {
      is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value
    };

    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_reduce");

    const size_t pool_reduce_bytes =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

    m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                                   ,
                                   0  // team_shared_bytes
                                   ,
                                   0  // thread_local_bytes
    );

    const int pool_size = OpenMP::impl_thread_pool_size();
#pragma omp parallel num_threads(pool_size)
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      data.set_work_partition(m_policy.end() - m_policy.begin(),
                              m_policy.chunk_size());

      if (is_dynamic) {
        // Make sure work partition is set before stealing
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      reference_type update =
          ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                          data.pool_reduce_local());

      std::pair<int64_t, int64_t> range(0, 0);

      do {
        range = is_dynamic ? data.get_work_stealing_chunk()
                           : data.get_work_partition();

        ParallelReduce::template exec_range<WorkTag>(
            m_functor, range.first + m_policy.begin(),
            range.second + m_policy.begin(), update);

      } while (is_dynamic && 0 <= range.first);
    }

    // Reduction:

    const pointer_type ptr =
        pointer_type(m_instance->get_thread_data(0)->pool_reduce_local());

    for (int i = 1; i < pool_size; ++i) {
      ValueJoin::join(ReducerConditional::select(m_functor, m_reducer), ptr,
                      m_instance->get_thread_data(i)->pool_reduce_local());
    }

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);

    if (m_result_ptr) {
      const int n = Analysis::value_count(
          ReducerConditional::select(m_functor, m_reducer));

      for (int j = 0; j < n; ++j) {
        m_result_ptr[j] = ptr[j];
      }
    }
  }

  //----------------------------------------

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, Policy arg_policy,
      const ViewType& arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_view.data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
};

// MDRangePolicy impl
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::OpenMP> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using WorkTag   = typename MDRangePolicy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                   MDRangePolicy, FunctorType>;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using value_type     = typename Analysis::value_type;
  using reference_type = typename Analysis::reference_type;

  using iterate_type =
      typename Kokkos::Impl::HostIterateTile<MDRangePolicy, FunctorType,
                                             WorkTag, reference_type>;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
                          // ).set_chunk_size(1) in ctor
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  inline static void exec_range(const MDRangePolicy& mdr_policy,
                                const FunctorType& functor, const Member ibeg,
                                const Member iend, reference_type update) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      iterate_type(mdr_policy, functor, update)(iwork);
    }
  }

 public:
  inline void execute() const {
    enum {
      is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value
    };

    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_reduce");

    const size_t pool_reduce_bytes =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

    m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                                   ,
                                   0  // team_shared_bytes
                                   ,
                                   0  // thread_local_bytes
    );

    const int pool_size = OpenMP::impl_thread_pool_size();
#pragma omp parallel num_threads(pool_size)
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      data.set_work_partition(m_policy.end() - m_policy.begin(),
                              m_policy.chunk_size());

      if (is_dynamic) {
        // Make sure work partition is set before stealing
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      reference_type update =
          ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                          data.pool_reduce_local());

      std::pair<int64_t, int64_t> range(0, 0);

      do {
        range = is_dynamic ? data.get_work_stealing_chunk()
                           : data.get_work_partition();

        ParallelReduce::exec_range(m_mdr_policy, m_functor,
                                   range.first + m_policy.begin(),
                                   range.second + m_policy.begin(), update);

      } while (is_dynamic && 0 <= range.first);
    }
    // END #pragma omp parallel

    // Reduction:

    const pointer_type ptr =
        pointer_type(m_instance->get_thread_data(0)->pool_reduce_local());

    for (int i = 1; i < pool_size; ++i) {
      ValueJoin::join(ReducerConditional::select(m_functor, m_reducer), ptr,
                      m_instance->get_thread_data(i)->pool_reduce_local());
    }

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);

    if (m_result_ptr) {
      const int n = Analysis::value_count(
          ReducerConditional::select(m_functor, m_reducer));

      for (int j = 0; j < n; ++j) {
        m_result_ptr[j] = ptr[j];
      }
    }
  }

  //----------------------------------------

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, MDRangePolicy arg_policy,
      const ViewType& arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(InvalidType()),
        m_result_ptr(arg_view.data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }

  inline ParallelReduce(const FunctorType& arg_functor,
                        MDRangePolicy arg_policy, const ReducerType& reducer)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                   Kokkos::OpenMP> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
  using ValueOps  = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update, const bool final) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork, update, final);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update, const bool final) {
    const TagType t{};
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork, update, final);
    }
  }

 public:
  inline void execute() const {
    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_scan");

    const int value_count          = Analysis::value_count(m_functor);
    const size_t pool_reduce_bytes = 2 * Analysis::value_size(m_functor);

    m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                                   ,
                                   0  // team_shared_bytes
                                   ,
                                   0  // thread_local_bytes
    );

#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      const WorkRange range(m_policy, omp_get_thread_num(),
                            omp_get_num_threads());

      reference_type update_sum =
          ValueInit::init(m_functor, data.pool_reduce_local());

      ParallelScan::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_sum, false);

      if (data.pool_rendezvous()) {
        pointer_type ptr_prev = nullptr;

        const int n = omp_get_num_threads();

        for (int i = 0; i < n; ++i) {
          pointer_type ptr =
              (pointer_type)data.pool_member(i)->pool_reduce_local();

          if (i) {
            for (int j = 0; j < value_count; ++j) {
              ptr[j + value_count] = ptr_prev[j + value_count];
            }
            ValueJoin::join(m_functor, ptr + value_count, ptr_prev);
          } else {
            ValueInit::init(m_functor, ptr + value_count);
          }

          ptr_prev = ptr;
        }

        data.pool_rendezvous_release();
      }

      reference_type update_base = ValueOps::reference(
          ((pointer_type)data.pool_reduce_local()) + value_count);

      ParallelScan::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_base, true);
    }
  }

  //----------------------------------------

  inline ParallelScan(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {}

  //----------------------------------------
};

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::OpenMP> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
  using ValueOps  = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  ReturnType& m_returnvalue;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update, const bool final) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork, update, final);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update, const bool final) {
    const TagType t{};
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork, update, final);
    }
  }

 public:
  inline void execute() const {
    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_scan");

    const int value_count          = Analysis::value_count(m_functor);
    const size_t pool_reduce_bytes = 2 * Analysis::value_size(m_functor);

    m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                                   ,
                                   0  // team_shared_bytes
                                   ,
                                   0  // thread_local_bytes
    );

#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      const WorkRange range(m_policy, omp_get_thread_num(),
                            omp_get_num_threads());
      reference_type update_sum =
          ValueInit::init(m_functor, data.pool_reduce_local());

      ParallelScanWithTotal::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_sum, false);

      if (data.pool_rendezvous()) {
        pointer_type ptr_prev = nullptr;

        const int n = omp_get_num_threads();

        for (int i = 0; i < n; ++i) {
          pointer_type ptr =
              (pointer_type)data.pool_member(i)->pool_reduce_local();

          if (i) {
            for (int j = 0; j < value_count; ++j) {
              ptr[j + value_count] = ptr_prev[j + value_count];
            }
            ValueJoin::join(m_functor, ptr + value_count, ptr_prev);
          } else {
            ValueInit::init(m_functor, ptr + value_count);
          }

          ptr_prev = ptr;
        }

        data.pool_rendezvous_release();
      }

      reference_type update_base = ValueOps::reference(
          ((pointer_type)data.pool_reduce_local()) + value_count);

      ParallelScanWithTotal::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_base, true);

      if (omp_get_thread_num() == omp_get_num_threads() - 1) {
        m_returnvalue = update_base;
      }
    }
  }

  //----------------------------------------

  inline ParallelScanWithTotal(const FunctorType& arg_functor,
                               const Policy& arg_policy,
                               ReturnType& arg_returnvalue)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_returnvalue(arg_returnvalue) {}

  //----------------------------------------
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::OpenMP> {
 private:
  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::OpenMP, Properties...>;
  using WorkTag  = typename Policy::work_tag;
  using SchedTag = typename Policy::schedule_type::type;
  using Member   = typename Policy::member_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const int m_shmem_size;

  template <class TagType>
  inline static
      typename std::enable_if<(std::is_same<TagType, void>::value)>::type
      exec_team(const FunctorType& functor, HostThreadTeamData& data,
                const int league_rank_begin, const int league_rank_end,
                const int league_size) {
    for (int r = league_rank_begin; r < league_rank_end;) {
      functor(Member(data, r, league_size));

      if (++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if (data.team_rendezvous()) {
          data.team_rendezvous_release();
        }
      }
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<(!std::is_same<TagType, void>::value)>::type
      exec_team(const FunctorType& functor, HostThreadTeamData& data,
                const int league_rank_begin, const int league_rank_end,
                const int league_size) {
    const TagType t{};

    for (int r = league_rank_begin; r < league_rank_end;) {
      functor(t, Member(data, r, league_size));

      if (++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if (data.team_rendezvous()) {
          data.team_rendezvous_release();
        }
      }
    }
  }

 public:
  inline void execute() const {
    enum { is_dynamic = std::is_same<SchedTag, Kokkos::Dynamic>::value };

    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_for");

    const size_t pool_reduce_size  = 0;  // Never shrinks
    const size_t team_reduce_size  = TEAM_REDUCE_SIZE * m_policy.team_size();
    const size_t team_shared_size  = m_shmem_size + m_policy.scratch_size(1);
    const size_t thread_local_size = 0;  // Never shrinks

    m_instance->resize_thread_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      const int active = data.organize_team(m_policy.team_size());

      if (active) {
        data.set_work_partition(
            m_policy.league_size(),
            (0 < m_policy.chunk_size() ? m_policy.chunk_size()
                                       : m_policy.team_iter()));
      }

      if (is_dynamic) {
        // Must synchronize to make sure each team has set its
        // partition before beginning the work stealing loop.
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      if (active) {
        std::pair<int64_t, int64_t> range(0, 0);

        do {
          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelFor::template exec_team<WorkTag>(m_functor, data, range.first,
                                                   range.second,
                                                   m_policy.league_size());

        } while (is_dynamic && 0 <= range.first);
      }

      data.disband_team();
    }
  }

  inline ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_shmem_size(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         arg_functor, arg_policy.team_size())) {}
};

//----------------------------------------------------------------------------

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::OpenMP> {
 private:
  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::OpenMP, Properties...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  using WorkTag  = typename Policy::work_tag;
  using SchedTag = typename Policy::schedule_type::type;
  using Member   = typename Policy::member_type;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;

  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;
  const int m_shmem_size;

  template <class TagType>
  inline static
      typename std::enable_if<(std::is_same<TagType, void>::value)>::type
      exec_team(const FunctorType& functor, HostThreadTeamData& data,
                reference_type& update, const int league_rank_begin,
                const int league_rank_end, const int league_size) {
    for (int r = league_rank_begin; r < league_rank_end;) {
      functor(Member(data, r, league_size), update);

      if (++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if (data.team_rendezvous()) {
          data.team_rendezvous_release();
        }
      }
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<(!std::is_same<TagType, void>::value)>::type
      exec_team(const FunctorType& functor, HostThreadTeamData& data,
                reference_type& update, const int league_rank_begin,
                const int league_rank_end, const int league_size) {
    const TagType t{};

    for (int r = league_rank_begin; r < league_rank_end;) {
      functor(t, Member(data, r, league_size), update);

      if (++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if (data.team_rendezvous()) {
          data.team_rendezvous_release();
        }
      }
    }
  }

 public:
  inline void execute() const {
    enum { is_dynamic = std::is_same<SchedTag, Kokkos::Dynamic>::value };

    if (m_policy.league_size() * m_policy.team_size() == 0) {
      if (m_result_ptr) {
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                        m_result_ptr);
        Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
            ReducerConditional::select(m_functor, m_reducer), m_result_ptr);
      }
      return;
    }
    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_reduce");

    const size_t pool_reduce_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

    const size_t team_reduce_size  = TEAM_REDUCE_SIZE * m_policy.team_size();
    const size_t team_shared_size  = m_shmem_size + m_policy.scratch_size(1);
    const size_t thread_local_size = 0;  // Never shrinks

    m_instance->resize_thread_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    const int pool_size = OpenMP::impl_thread_pool_size();
#pragma omp parallel num_threads(pool_size)
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      const int active = data.organize_team(m_policy.team_size());

      if (active) {
        data.set_work_partition(
            m_policy.league_size(),
            (0 < m_policy.chunk_size() ? m_policy.chunk_size()
                                       : m_policy.team_iter()));
      }

      if (is_dynamic) {
        // Must synchronize to make sure each team has set its
        // partition before beginning the work stealing loop.
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      if (active) {
        reference_type update =
            ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                            data.pool_reduce_local());

        std::pair<int64_t, int64_t> range(0, 0);

        do {
          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelReduce::template exec_team<WorkTag>(m_functor, data, update,
                                                      range.first, range.second,
                                                      m_policy.league_size());

        } while (is_dynamic && 0 <= range.first);
      } else {
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                        data.pool_reduce_local());
      }

      data.disband_team();

      //  This thread has updated 'pool_reduce_local()' with its
      //  contributions to the reduction.  The parallel region is
      //  about to terminate and the master thread will load and
      //  reduce each 'pool_reduce_local()' contribution.
      //  Must 'memory_fence()' to guarantee that storing the update to
      //  'pool_reduce_local()' will complete before this thread
      //  exits the parallel region.

      memory_fence();
    }

    // Reduction:

    const pointer_type ptr =
        pointer_type(m_instance->get_thread_data(0)->pool_reduce_local());

    for (int i = 1; i < pool_size; ++i) {
      ValueJoin::join(ReducerConditional::select(m_functor, m_reducer), ptr,
                      m_instance->get_thread_data(i)->pool_reduce_local());
    }

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);

    if (m_result_ptr) {
      const int n = Analysis::value_count(
          ReducerConditional::select(m_functor, m_reducer));

      for (int j = 0; j < n; ++j) {
        m_result_ptr[j] = ptr[j];
      }
    }
  }

  //----------------------------------------

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const ViewType& arg_result,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_shmem_size(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         arg_functor, arg_policy.team_size())) {}

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_shmem_size(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         arg_functor, arg_policy.team_size())) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                            , Kokkos::HostSpace >::value
    , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
    );*/
  }
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
#endif /* KOKKOS_OPENMP_PARALLEL_HPP */
