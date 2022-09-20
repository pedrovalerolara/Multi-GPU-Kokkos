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
#include <mpi.h>

#define SQ(x) ((x) * (x))

int main( int argc, char* argv[] )
{
  //int N = 83400;//167000;//340000;//670000;//1340000;//2670000;//5340000;//16700000;//21340000;//42670000;//85350000;
  int N_vector[11] = {83400, 167000, 340000, 670000, 1340000, 2670000, 5340000, 16700000, 21340000, 42670000, 85350000};

  MPI_Init(&argc,&argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Status status;

  printf("World rank = %d\n", world_rank);
 
  int N = N_vector[3];

  Kokkos::initialize( argc, argv );
  {

  struct timeval start, end;
  double ttime;

  auto X  = static_cast<double*>(Kokkos::kokkos_malloc<>(N * sizeof(double)));
  auto Y  = static_cast<double*>(Kokkos::kokkos_malloc<>(N * sizeof(double)));
  auto stencil_left  = static_cast<double*>(Kokkos::kokkos_malloc<>(1 * sizeof(double)));
  auto stencil_right  = static_cast<double*>(Kokkos::kokkos_malloc<>(1 * sizeof(double)));

  Kokkos::parallel_for( "stencil_init", N, KOKKOS_LAMBDA ( int n )
  {
    X[n] = 2.0;
    Y[n] = 2.0;
  });

  Kokkos::fence();

  gettimeofday( &start, NULL );
  for (int i = 0; i < 10; i++ )
  {
    //Communication to/from the right
    if (world_rank%2 == 0)
    {
      MPI_Send(&X[N-2], 1, MPI_DOUBLE, world_rank+1, 100, MPI_COMM_WORLD);
    }
    else if (world_rank%2 == 1)
    {
      MPI_Recv(&X[0], 1, MPI_DOUBLE, world_rank-1, 100, MPI_COMM_WORLD, &status);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  
    //Communication to/from the left
    if (world_rank%2 == 0)
    {
      MPI_Recv(&X[N-1], 1, MPI_DOUBLE, world_rank+1, 100, MPI_COMM_WORLD, &status);
    }
    else if (world_rank%2 == 1)
    {
      MPI_Send(&X[1], 1, MPI_DOUBLE, world_rank-1, 100, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  
    Kokkos::parallel_for( "stencil_compute", N, KOKKOS_LAMBDA ( int n )
    {
      double alpha = 2.0;
      if (n > 0 && n < N)
        Y[n] = alpha * X[n] + X[n-1] + X[n+1];
    });

    Kokkos::fence();
  }  
  
  gettimeofday( &end, NULL );
  ttime = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);

  printf( "Stencil time ( %e s )\n", ttime/10.0 );

  Kokkos::kokkos_free<>(X);
  Kokkos::kokkos_free<>(Y);

  }
  
  Kokkos::finalize();

  MPI_Finalize();

  return 0;
}


