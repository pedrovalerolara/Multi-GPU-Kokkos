/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

#define SQ(x) ((x) * (x))

int main( int argc, char* argv[] )
{
  int N = 128000;       
  int nrepeat = 1;
  float alpha = 2.0;

  Kokkos::initialize( argc, argv );
  {

  struct timeval start_cpu, start_het, start_gpu, end_cpu, end_het, end_gpu;

  auto X_cpu  = static_cast<float*>(Kokkos::kokkos_malloc<>(N * sizeof(float)));
  auto Y_cpu  = static_cast<float*>(Kokkos::kokkos_malloc<>(N * sizeof(float)));
  auto X_het  = static_cast<float*>(Kokkos::kokkos_malloc<>(N * sizeof(float)));
  auto Y_het  = static_cast<float*>(Kokkos::kokkos_malloc<>(N * sizeof(float)));
  
  Kokkos::parallel_for( "axpy_cpu", N, KOKKOS_LAMBDA ( int n )
  {
    X_cpu[n] = 2.0;
    Y_cpu[n] = 2.0;
    X_het[n] = 2.0;
    Y_het[n] = 2.0;
  });

  double time_cpu;
  double ttime_cpu;

  for (int i = 0; i < 20; i++ )
  {
  Kokkos::Timer timer_cpu;
  gettimeofday( &start_cpu, NULL );
  
  Kokkos::parallel_for( "axpy_cpu", N, KOKKOS_LAMBDA ( int n )
  {
    Y_cpu[n] += alpha * X_cpu[n];
  });
 
  gettimeofday( &end_cpu, NULL );
  time_cpu = timer_cpu.seconds();
  ttime_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
				 - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
 
  printf( "time cpu ( %g s ) ttime cpu ( %e s )\n",
          time_cpu, ttime_cpu );
  }

  /* 
  double time_het;
  double ttime_het;
  for (int i = 0; i < 20; i++ )
  {
  Kokkos::Timer timer_het;
  gettimeofday( &start_het, NULL );
 
  Kokkos::hetparallel_for_auto( "axpy_het", 2 * sizeof(float), 2, N, KOKKOS_LAMBDA ( int n ) 
  {
      Y_het[n] += alpha * X_het[n];
  });
  
  gettimeofday( &end_het, NULL );
  time_het = timer_het.seconds();
  ttime_het = ( double ) (((end_het.tv_sec * 1e6 + end_het.tv_usec)
		 - (start_het.tv_sec * 1e6 + start_het.tv_usec)) / 1e6);

  printf( "time het ( %g s ) ttime het ( %e s )\n",
          time_het, ttime_het );
  }
  */

  /* 
  double time_gpu;
  double ttime_gpu;
  for (int i = 0; i < 20; i++ )
  {
    Kokkos::Timer timer_gpu;
    gettimeofday( &start_gpu, NULL );
    
    #pragma omp target teams distribute parallel for map(to:X_het[0:N]) map(tofrom:Y_het[0:N]) 
    //#pragma omp target map(to:X_het[0:N]) map(tofrom:Y_het[0:N]) 
    //#pragma omp parallel for 
    for (int n = 0; n < N; n++)
    { 
      Y_het[n] = alpha * X_het[n] + Y_het[n];
    }
 
    gettimeofday( &end_gpu, NULL );
    time_gpu = timer_gpu.seconds();
    ttime_gpu = ( double ) (((end_gpu.tv_sec * 1e6 + end_gpu.tv_usec)
				 - (start_gpu.tv_sec * 1e6 + start_gpu.tv_usec)) / 1e6);
 
    printf( "time gpu omp ( %g s ) ttime gpu omp ( %e s )\n",
            time_gpu, ttime_gpu );
  }
  */

  /*
  double time_gpu;
  double ttime_gpu;
  //#pragma omp target map(to:X_het[0:N]) map(tofrom:Y_het[0:N]) 
    #pragma omp target enter data map(to:X_het[0:N]) map(to:Y_het[0:N]) 
    for (int i = 0; i < 20; i++ )
    {
      Kokkos::Timer timer_gpu;
      gettimeofday( &start_gpu, NULL );
    
      #pragma omp target teams distribute parallel for 
      //#pragma omp parallel for 
      for (int n = 0; n < N; n++)
      {   
        Y_het[n] = alpha * X_het[n] + Y_het[n];
      }
 
      gettimeofday( &end_gpu, NULL );
      time_gpu = timer_gpu.seconds();
      ttime_gpu = ( double ) (((end_gpu.tv_sec * 1e6 + end_gpu.tv_usec)
				 - (start_gpu.tv_sec * 1e6 + start_gpu.tv_usec)) / 1e6);
 
      printf( "time gpu omp ( %g s ) ttime gpu omp ( %e s )\n",
              time_gpu, ttime_gpu );
    }
  //}
    #pragma omp target exit data map(from:Y_het[0:N]) 

  for ( int i = 0; i < N; i++ )
  {
    if (Y_het[i] != Y_cpu[i] )
      printf("Error\n");
  }
  */
  
  Kokkos::kokkos_free<>(X_cpu);
  Kokkos::kokkos_free<>(X_het);
  Kokkos::kokkos_free<>(Y_cpu);
  Kokkos::kokkos_free<>(Y_het);
  printf( "Hey\n" );

  }
  printf( "Heyy\n" );
  Kokkos::finalize();
  printf( "Heyyy\n" );

  return 0;
}
