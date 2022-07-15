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
  int N = 512000000;

  Kokkos::initialize( argc, argv );
  {

  struct timeval start, end;
  double ttime;

  auto X  = static_cast<double*>(Kokkos::kokkos_malloc<>(N * sizeof(double)));
  auto Y  = static_cast<double*>(Kokkos::kokkos_malloc<>(N * sizeof(double)));

  Kokkos::parallel_for( "axpy_init", N, KOKKOS_LAMBDA ( int n )
  {
    X[n] = 2.0;
    Y[n] = 2.0;
  });

  Kokkos::fence();

  gettimeofday( &start, NULL );
  for (int i = 0; i < 10; i++ )
  {
    Kokkos::parallel_for( "axpy_compute", N, KOKKOS_LAMBDA ( int n )
    {
      double alpha = 2.0;
      Y[n] = alpha * X[n] + Y[n];
    });

    Kokkos::fence();
  }

  gettimeofday( &end, NULL );
  ttime = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);

  printf( "AXPY time single gpu ( %e s )\n", ttime/10.0 );

  Kokkos::dKokkos_multi_dev_mem* X_multi[Kokkos::MAX_DEV];
  Kokkos::dKokkos_multi_dev_mem* Y_multi[Kokkos::MAX_DEV];
  Kokkos::dkokkos_malloc_multi_dev<>(X_multi, N * sizeof(double));
  Kokkos::dkokkos_malloc_multi_dev<>(Y_multi, N * sizeof(double));

  Kokkos::parallel_for_multi_dev( "axpy_init", N, KOKKOS_LAMBDA ( int i, int n )
  {
    X_multi[i]->array[n] = 2.0;
    Y_multi[i]->array[n] = 2.0;
  });

  gettimeofday( &start, NULL );
  for (int i = 0; i < 10; i++ )
  {
    Kokkos::parallel_for_multi_dev( "axpy_init", N, KOKKOS_LAMBDA ( int i, int n )
    {
        double alpha = 2.0;
        Y_multi[i]->array[n] += alpha * X_multi[i]->array[n];
        //printf("Dev id = %d, input = %1.2f, output = %1.2f\n",i,X_multi[i]->array[n], Y_multi[i]->array[n]);
    });

  }
  gettimeofday( &end, NULL );
  ttime = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);

  printf( "AXPY time multi gpu ( %e s )\n", ttime/10.0 );

  int num_dev = Kokkos::kokkos_multi_dev_auto<>(2 * N * sizeof(double));
  printf ( "Num dev used = %d\n", num_dev );

  Kokkos::dKokkos_multi_dev_mem* X_multi_auto[Kokkos::MAX_DEV];
  Kokkos::dKokkos_multi_dev_mem* Y_multi_auto[Kokkos::MAX_DEV];
  Kokkos::dkokkos_malloc_multi_dev_auto<>(X_multi_auto, N * sizeof(double), num_dev);
  Kokkos::dkokkos_malloc_multi_dev_auto<>(Y_multi_auto, N * sizeof(double), num_dev);

  Kokkos::parallel_for_multi_dev_auto( "axpy_init", num_dev, N, KOKKOS_LAMBDA ( int i, int n )
  {
    X_multi_auto[i]->array[n] = 2.0;
    Y_multi_auto[i]->array[n] = 2.0;
  });

  gettimeofday( &start, NULL );
  for (int i = 0; i < 10; i++ )
  {
    Kokkos::parallel_for_multi_dev_auto( "axpy_init", num_dev, N, KOKKOS_LAMBDA ( int i, int n )
    {
        double alpha = 2.0;
        Y_multi_auto[i]->array[n] += alpha * X_multi_auto[i]->array[n];
        //printf("Dev id = %d, input = %1.2f, output = %1.2f\n",i,X_multi_auto[i]->array[n], Y_multi_auto[i]->array[n]);
    });

  }
  gettimeofday( &end, NULL );
  ttime = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);

  printf( "AXPY time multi gpu auto ( %e s )\n", ttime/10.0 );

  Kokkos::kokkos_free<>(X);
  Kokkos::kokkos_free<>(Y);

  }
  Kokkos::finalize();

  return 0;
}


