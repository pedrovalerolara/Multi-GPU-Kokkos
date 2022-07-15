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
  int M = 10000;       
  int N = 10000;       
  int nrepeat = 1;
  double alpha = 2.0;
  double beta = 3.0;        

  Kokkos::initialize( argc, argv );
  {

  struct timeval start_cpu, start_het, end_cpu, end_het;

  auto A_cpu  = static_cast<double*>(Kokkos::kokkos_malloc<>(M * N * sizeof(double)));
  auto X_cpu  = static_cast<double*>(Kokkos::kokkos_malloc<>(N * sizeof(double)));
  auto Y_cpu  = static_cast<double*>(Kokkos::kokkos_malloc<>(M * sizeof(double)));
  auto A_het  = static_cast<double*>(Kokkos::kokkos_malloc<>(M * N * sizeof(double)));
  auto X_het  = static_cast<double*>(Kokkos::kokkos_malloc<>(N * sizeof(double)));
  auto Y_het  = static_cast<double*>(Kokkos::kokkos_malloc<>(M * sizeof(double)));
  
  for ( int i = 0; i < M * N; i++ )
  {
    A_cpu[i] = 5.0;
    A_het[i] = 5.0;
  }

  for ( int i = 0; i < N; i++ )
  {
    X_cpu[i] = 2.0;
    X_het[i] = 2.0;
  }
  
  for ( int i = 0; i < M; i++ )
  {
    Y_cpu[i] = 3.0;
    Y_het[i] = 3.0;
  }

  struct timeval start_iter_seq, start_iter_het, end_iter_seq, end_iter_het;
 
  double time_iter_seq;
  double ttime_iter_seq;

  for (int i = 0; i < 10; i++)
  { 
  Kokkos::Timer timer_iter_seq;
  gettimeofday( &start_iter_seq, NULL );
  
  for (int iter = 0; iter < 10; iter++)
  { 
    Kokkos::parallel_for( "dgemv_het", M, KOKKOS_LAMBDA ( int m ) 
    {
      for ( int n = 0; n < N; n++ ) 
        Y_het[m] = alpha * A_het[m * N + n] * X_het[n] + beta * Y_het[m];
    });
  }
  
  gettimeofday( &end_iter_seq, NULL );
  time_iter_seq = timer_iter_seq.seconds();
  ttime_iter_seq = ( double ) (((end_iter_seq.tv_sec * 1e6 + end_iter_seq.tv_usec)
				 - (start_iter_seq.tv_sec * 1e6 + start_iter_seq.tv_usec)) / 1e6);

  printf( "time iter seq ( %g s ) ttime iter seq ( %e s )\n",
          time_iter_seq, ttime_iter_seq );
  }

  double time_iter_het;
  double ttime_iter_het;
  for (int i = 0; i < 10; i++)
  { 
  Kokkos::Timer timer_iter_het;
  gettimeofday( &start_iter_het, NULL );
  Kokkos::hetparallel_for_iter_auto( "dgemv_het", 0, 10, N * sizeof(double), 4 * N, M, KOKKOS_LAMBDA ( int m ) 
  {
    for ( int n = 0; n < N; n++ ) 
      Y_het[m] = alpha * A_het[m * N + n] * X_het[n] + beta * Y_het[m];
  });
  
  gettimeofday( &end_iter_het, NULL );
  double time_iter_het = timer_iter_het.seconds();
  double ttime_iter_het = ( double ) (((end_iter_het.tv_sec * 1e6 + end_iter_het.tv_usec)
				 - (start_iter_het.tv_sec * 1e6 + start_iter_het.tv_usec)) / 1e6);

  printf( "time iter het ( %g s ) ttime iter het ( %e s )\n",
          time_iter_het, ttime_iter_het );

  }

  Kokkos::kokkos_free<>(A_cpu);
  Kokkos::kokkos_free<>(A_het);
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
