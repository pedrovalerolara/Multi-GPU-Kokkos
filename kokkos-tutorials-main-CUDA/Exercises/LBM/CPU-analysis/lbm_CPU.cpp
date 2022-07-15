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
  int Nx = 10000;       
  int Ny = 1000;
  int Nlatt = 9;
  int nrepeat = 1;        

  Kokkos::initialize( argc, argv );
  {

  #ifdef KOKKOS_ENABLE_CUDA
  #define MemSpace Kokkos::CudaSpace
  #endif
  #ifdef KOKKOS_ENABLE_HIP
  #define MemSpace Kokkos::Experimental::HIPSpace
  #endif
  #ifdef KOKKOS_ENABLE_OPENMPTARGET
  #define MemSpace Kokkos::OpenMPTargetSpace
  #endif

  #ifndef MemSpace
  #define MemSpace Kokkos::HostSpace
  #endif

  using ExecSpace = MemSpace::execution_space;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;
  struct timeval start, end;
  //typedef Kokkos::MDRangePolicy< Kokkos::Rank<3> > mdrange_policy;
  //typedef Kokkos::MDRangePolicy< Kokkos::Rank<2> > mdrange_policy;
  typedef Kokkos::MDRangePolicy< Kokkos::Rank<1> > mdrange_policy;

  typedef Kokkos::View<double*> lattice;
  lattice cx("vector-horizontal", 9), cy("vector-vertical", 9), weight("weight", 9);
  typedef Kokkos::View<double*> grid;
  grid u("horizontal-speed", Nx*Ny), v("vertical-speed", Nx*Ny), p("density", Nx*Ny);
  typedef Kokkos::View<double*> latticeforce; 
  latticeforce f1("f1-lattice", Nx*Ny*9), f2("f2-lattice", Nx*Ny*9);

  Kokkos::parallel_for( "init", Nx * Ny * 9, KOKKOS_LAMBDA ( int x )
  {
      f1(x) = 100.0;
      f2(x) = 100.0;
  });
 
  Kokkos::parallel_for( "init", Nx * Ny, KOKKOS_LAMBDA ( int x )
  {
      u(x) = 10.0;
      v(x) = 10.0;
      p(x) = 10.0;
  });

  Kokkos::parallel_for( "init-weight", 9, KOKKOS_LAMBDA ( int x ) 
  {
      if ( x == 0 ) 
      {  
          weight(x) =  4./9.;
      }
      else if ( x < 5 )
      {
          weight(x) =  1./9.;
      }
      else
      {
          weight(x) =  1./36.;
      }
  });
  
  Kokkos::parallel_for( "init-cx", 9, KOKKOS_LAMBDA ( int x ) 
  {
      if ( x == 0 ) 
      {  
          cx(x) = 0.;
      }
      else if ( x == 1 )
      {
          cx(x) = 1.;
      }
      else if ( x == 2 )
      {
          cx(x) = 0.;
      }
      else if ( x == 3 )
      {
          cx(x) = -1.;
      }
      else if ( x == 4 )
      {
          cx(x) = 0.;
      }
      else if ( x == 5 )
      {
          cx(x) = 1.;
      }
      else if ( x == 6 )
      {
          cx(x) = -1.;
      }
      else if ( x == 7 )
      {
          cx(x) = -1.;
      }
      else
      {
          cx(x) = 1.;
      }
  });

  Kokkos::parallel_for( "init-cy", 9, KOKKOS_LAMBDA ( int x ) 
  {
      if ( x == 0 ) 
      {  
          cy(x) = 0.;
      }
      else if ( x == 1 )
      {
          cy(x) = 0.;
      }
      else if ( x == 2 )
      {
          cy(x) = 1.;
      }
      else if ( x == 3 )
      {
          cy(x) = 0.;
      }
      else if ( x == 4 )
      {
          cy(x) = -1.;
      }
      else if ( x == 5 )
      {
          cy(x) = 1.;
      }
      else if ( x == 6 )
      {
          cy(x) = 1.;
      }
      else if ( x == 7 )
      {
          cy(x) = -1.;
      }
      else
      {
          cy(x) = -1.;
      }
  });
 
  /*
  weight(0) =  4./9.;
  weight(1) =  1./9.;
  weight(2) =  1./9.;
  weight(3) =  1./9.;
  weight(4) =  1./9.;
  weight(5) =  1./36.;
  weight(6) =  1./36.;
  weight(7) =  1./36.;
  weight(8) =  1./36.;
  */

  /*
  cx(0) =  0.0; 
  cx(1) =  1.; 
  cx(2) =  0.; 
  cx(3) = -1.; 
  cx(4) =  0.; 
  cx(5) =  1.; 
  cx(6) = -1.; 
  cx(7) = -1.; 
  cx(8) =  1.; 

  cy(0) =  0.; 
  cy(1) =  0.; 
  cy(2) =  1.; 
  cy(3) =  0.; 
  cy(4) = -1.; 
  cy(5) =  1.; 
  cy(6) =  1.; 
  cy(7) = -1.; 
  cy(8) = -1.; 
  */

  int output = 100;
  double omega = 3.0;

  // Timer products.
  Kokkos::Timer timer;
  gettimeofday( &start, NULL );

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    
    //Kokkos::parallel_for( "LBM-collide-stream", mdrange_policy( {1,1,0}, {Ny-1,Nx-1,Nlatt}), KOKKOS_LAMBDA ( const int y, const int x, const int k ) 
    //Kokkos::parallel_for( "LBM-collide-stream", mdrange_policy( {1,1}, {Ny-1,Nx-1}), KOKKOS_LAMBDA ( const int y, const int x ) 
    //Kokkos::parallel_for( "LBM-collide-stream", mdrange_policy( {1}, {Ny-1}), KOKKOS_LAMBDA ( const int y ) 
    //{   
    Kokkos::parallel_for( "LBM-collide-stream", range_policy( 0, Ny ), KOKKOS_LAMBDA ( int y ) 
    {   
        int ind, new_x, new_y, new_ind;
        double cu, feq;
        
        for ( int x = 1; x < Nx - 1; x++ )
        { 

            ind = ( y * Nx ) + x;
	    
            for ( int k = 0; k < 9; k++ )
            {
	        //Collision
	        cu = 3.0 * ( cx(k) * u(ind) + cy(k) * v(ind) );
                feq = p(ind) * weight(k) * ( 1.0 + cu + ( 1.0/2.0 * SQ( cu ) ) - ( 3.0/2.0 * ( SQ( u(ind) ) + SQ( v(ind) ) ) ) );
                //Stream 
	        //Coalesced accesses
                new_x = x + cx(k);
                new_y = x + cy(k);
                new_ind = new_y * Nx + new_x;              
 
	        if( repeat % 2 == 0 )
                {
	            f2( k * ( Nx * Ny) + new_ind ) = f1( k * ( Nx * Ny) + ind ) - omega * ( f1( k * ( Nx * Ny) + ind ) - feq );
	        }
	        else
                {
	            f1( k * ( Nx * Ny) + new_ind ) = f2( k * ( Nx * Ny) + ind ) - omega * ( f2( k * ( Nx * Ny) + ind ) - feq );
	        }
            }
        }
    });

    //Kokkos::parallel_for( "LBM-macro", mdrange_policy( {1,1,0}, {Ny-1,Nx-1,Nlatt} ), KOKKOS_LAMBDA ( const int y, const int x, const int k ) 
    //Kokkos::parallel_for( "LBM-macro", mdrange_policy( {1,1}, {Ny-1,Nx-1} ), KOKKOS_LAMBDA ( const int y, const int x ) 
    //Kokkos::parallel_for( "LBM-macro", mdrange_policy( {1}, {Ny-1} ), KOKKOS_LAMBDA ( const int y ) 
    //{   
    Kokkos::parallel_for( "LBM-macro", range_policy( 0, Ny ), KOKKOS_LAMBDA ( int y ) 
    {   
        int ind;
        double pp, uu, vv;
        
        for ( int x = 1; x < Nx - 1; x++ )
        { 

            ind = ( y * Nx ) + x;
	    pp = 0.0;
	    uu = 0.0;
	    vv = 0.0;
	
            //Macro
	    for ( int k = 0; k < 9; k++ )
            {
		if( repeat % 2 == 0 )
                {
		    pp += f1( k * ( Nx * Ny ) + ind );
	            uu += cx(k) * f1( k * ( Nx * Ny ) + ind ); 
	            vv += cy(k) * f1( k * ( Nx * Ny ) + ind );
                }
		else
                {
		    pp += f2( k * ( Nx * Ny ) + ind );
	            uu += cx(k) * f2( k * ( Nx * Ny ) + ind ); 
	            vv += cy(k) * f2( k * ( Nx * Ny ) + ind );
		}
            }

	    uu /= pp;
	    vv /= pp;

	    u(ind) = uu;
	    v(ind) = vv;
            p(ind) = pp;
        }
    });
  }// End repeat loop

  gettimeofday( &end, NULL );
  double time = timer.seconds();
  double ttime = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
				 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e3);
  // Calculate bandwidth.
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( Nx * Ny * 9 * 2 ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "Nx = %d, Ny = %d, nrepeat = %d, problem( %g MB ) time( %g s ) bandwidth( %g GB/s ) MFLUPS ( %g )\n",
          Nx, Ny, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time, ( Nx * Ny * nrepeat ) / (ttime * 1000.0) );

  /*
  Kokkos::kokkos_free<>(cx);
  Kokkos::kokkos_free<>(cy);
  Kokkos::kokkos_free<>(u);
  Kokkos::kokkos_free<>(v);
  Kokkos::kokkos_free<>(p);
  Kokkos::kokkos_free<>(f1);
  Kokkos::kokkos_free<>(f2);
  */

  }
  Kokkos::finalize();

  return 0;
}
