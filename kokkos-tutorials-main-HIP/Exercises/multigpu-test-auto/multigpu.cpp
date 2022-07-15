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
  //int N = 128000000;       
  int N = 512;       
  //int N = 1310720;       
  //int N = 1073741824;       

  Kokkos::initialize( argc, argv );
  {

  struct timeval start_cpu, end_cpu;
  double ttime_cpu;

  auto X_cpu  = static_cast<float*>(Kokkos::kokkos_malloc<>(N * sizeof(float)));
  auto Y_cpu  = static_cast<float*>(Kokkos::kokkos_malloc<>(N * sizeof(float)));
  
  struct Kokkos::Kokkos_multi_dev_mem* X_multi;
  struct Kokkos::Kokkos_multi_dev_mem* Y_multi;
 
  X_multi = (struct Kokkos::Kokkos_multi_dev_mem*) malloc (sizeof(struct Kokkos::Kokkos_multi_dev_mem));
  Y_multi = (struct Kokkos::Kokkos_multi_dev_mem*) malloc (sizeof(struct Kokkos::Kokkos_multi_dev_mem));
  Kokkos::kokkos_malloc_multi_dev<>(X_multi, N * sizeof(float));
  Kokkos::kokkos_malloc_multi_dev<>(Y_multi, N * sizeof(float));
  
  int num_dev = Kokkos::kokkos_multi_dev_auto<>(2 * N * sizeof(float));
  printf("Num devices used = %d\n", num_dev);
  struct Kokkos::Kokkos_multi_dev_mem* X_multi_auto;
  struct Kokkos::Kokkos_multi_dev_mem* Y_multi_auto;
  X_multi_auto = (struct Kokkos::Kokkos_multi_dev_mem*) malloc (sizeof(struct Kokkos::Kokkos_multi_dev_mem));
  Y_multi_auto = (struct Kokkos::Kokkos_multi_dev_mem*) malloc (sizeof(struct Kokkos::Kokkos_multi_dev_mem));
  Kokkos::kokkos_malloc_multi_dev_auto<>(X_multi_auto, N * sizeof(float), num_dev);
  Kokkos::kokkos_malloc_multi_dev_auto<>(Y_multi_auto, N * sizeof(float), num_dev);
  //X_multi->space[0]  = static_cast<float*>(Kokkos::kokkos_malloc<>(N * sizeof(float)));
 
  //printf("Num devices = %d\n", X_multi->num_dev);
  //printf("Pointer = %p\n", &X_multi->space[0][0]);

  //for (int i = 0; i < 20; i++ )
  //{
    gettimeofday( &start_cpu, NULL );
  
    Kokkos::parallel_for( "axpy_init", N, KOKKOS_LAMBDA ( int n )
    {
      X_cpu[n] = 2.0;
      Y_cpu[n] = 2.0;
    });

    Kokkos::fence();

    gettimeofday( &end_cpu, NULL );
    ttime_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
				 - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
 
    //printf( "init time single gpu ( %e s )\n",
    //      ttime_cpu );
  //}

  //for (int i = 0; i < 20; i++ )
  //{
    gettimeofday( &start_cpu, NULL );
  
    Kokkos::parallel_for_multi_dev( "axpy_init", N, KOKKOS_LAMBDA ( int i, int n )
    {
      X_multi->space[i][n] = 2.0;
      Y_multi->space[i][n] = 2.0;
    });

    gettimeofday( &end_cpu, NULL );
    ttime_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
				 - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
 
    //printf( "init time multi gpu ( %e s )\n",
    //      ttime_cpu );
  //}

  
    Kokkos::parallel_for_multi_dev_auto( "axpy_init", num_dev, N, KOKKOS_LAMBDA ( int i, int n )
    {
      X_multi_auto->space[i][n] = 2.0;
      Y_multi_auto->space[i][n] = 2.0;
    });


  Kokkos::parallel_for( "axpy_init", N, KOKKOS_LAMBDA ( int n )
  {
    double alpha = 2.0;
    Y_cpu[n] = alpha * X_cpu[n] + Y_cpu[n];
  });

  Kokkos::fence();

  //for (int NN = N; NN >= 512; NN/=10)
  //{
  //  printf("NN = %d\n", NN);
  gettimeofday( &start_cpu, NULL );
  for (int i = 0; i < 20; i++ )
  {
    Kokkos::parallel_for( "axpy_init", N, KOKKOS_LAMBDA ( int n )
    {
      double alpha = 2.0;
      Y_cpu[n] = alpha * X_cpu[n] + Y_cpu[n];
    });

    Kokkos::fence();

  }
  gettimeofday( &end_cpu, NULL );
  ttime_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
				 - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
  ttime_cpu /= 20.0; 
 
  printf( "axpy time single gpu ( %e s )\n",
          ttime_cpu );

  Kokkos::parallel_for_multi_dev( "axpy_init", N, KOKKOS_LAMBDA ( int i, int n )
  {
    double alpha = 2.0;
    Y_multi->space[i][n] = alpha * X_multi->space[i][n] + Y_multi->space[i][n];
  });

  gettimeofday( &start_cpu, NULL );
  for (int i = 0; i < 20; i++ )
  {
    Kokkos::parallel_for_multi_dev( "axpy_init", N, KOKKOS_LAMBDA ( int i, int n )
    {
      double alpha = 2.0;
      Y_multi->space[i][n] = alpha * X_multi->space[i][n] + Y_multi->space[i][n];
    });
  }

  gettimeofday( &end_cpu, NULL );
  ttime_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
				 - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
  ttime_cpu /= 20.0; 

  printf( "axpy time multi gpu ( %e s )\n",
        ttime_cpu );

  Kokkos::parallel_for_multi_dev_auto( "axpy_init", num_dev, N, KOKKOS_LAMBDA ( int i, int n )
  {
    double alpha = 2.0;
    Y_multi_auto->space[i][n] = alpha * X_multi_auto->space[i][n] + Y_multi_auto->space[i][n];
  });

  gettimeofday( &start_cpu, NULL );
  for (int i = 0; i < 20; i++ )
  {
    Kokkos::parallel_for_multi_dev_auto( "axpy_init", num_dev, N, KOKKOS_LAMBDA ( int i, int n )
    {
      double alpha = 2.0;
      Y_multi_auto->space[i][n] = alpha * X_multi_auto->space[i][n] + Y_multi_auto->space[i][n];
    });
  }

  gettimeofday( &end_cpu, NULL );
  ttime_cpu = ( double ) (((end_cpu.tv_sec * 1e6 + end_cpu.tv_usec)
				 - (start_cpu.tv_sec * 1e6 + start_cpu.tv_usec)) / 1e6);
  ttime_cpu /= 20.0; 

  printf( "axpy time multi gpu auto ( %e s )\n",
        ttime_cpu );


  //}

  Kokkos::kokkos_free<>(X_cpu);
  Kokkos::kokkos_free<>(Y_cpu);

  }
  Kokkos::finalize();

  return 0;
}
