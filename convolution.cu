// Author: Karl Marrett
// adaptation from diannao basic kernels
// to cuda kernel for learning purposes
// MIT license

#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cassert>
#include "dnn.hpp"

using namespace std;

#define assertm(expr, msg) assert(((void) msg, expr));

// source from @talonmies
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
	  fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
   }
}

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)
const int output_neurons = 64;

void fill(VTYPE *arr, const uint64_t size) {
  for (int i=0; i < size; ++i) {
    arr[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
}

/*
std::pair<int,int> convolution_layer_blocked(
                              VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                              VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                              VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  int c1=0,c2=0;
  VTYPE sum[Nn]={0};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy/Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx/Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              for (int n = nn; n < nn + Tn; n++) {
                sum[n] = 0;
              }

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni -Ti+1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc=0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc+=sv*nv;
                      }
                      sum[n]+=sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++; 
          }
          yout++;
        }
      }
    }
  }
}
*/

__global__
void convolution_layer(VTYPE * synapse, VTYPE* neuron_i, VTYPE* neuron_n, int Nx,
                       int Ny, int Kx, int Ky, int Ni, int Nn) {
  // VTYPE sum[Nn]={0};
  VTYPE sum[output_neurons];

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Nx; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
	// reset sum for this window slide area
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
		// VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                VTYPE sv = synapse[ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i];
                  //            VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                VTYPE nv = neuron_i[(ky + y) * NXPAD * Ni + (kx + x) * Ni + i];
                // VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
	  // VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
          neuron_n[yout * (NXSCL * Nn) + xout * Nn + n] = sum[n] > 0 ? sum[n] : sum[n]/4;
          // neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}

__global__
void convolution_layer_parallel(VTYPE * synapse, VTYPE* neuron_i, VTYPE* neuron_n, int Nx,
                       int Ny, int Kx, int Ky, int Ni, int Nn) {
  // VTYPE sum[Nn]={0};
  VTYPE sum[output_neurons];

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Nx; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
	// reset sum for this window slide area
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
		// VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                VTYPE sv = synapse[ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i];
                  //            VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                VTYPE nv = neuron_i[(ky + y) * NXPAD * Ni + (kx + x) * Ni + i];
                // VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
	  // VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
          neuron_n[yout * (NXSCL * Nn) + xout * Nn + n] = sum[n] > 0 ? sum[n] : sum[n]/4;
          // neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}

int main(const int argc, const char** argv) {
  assertm(argc == 7, "6 args required usage: ./convolution nx ny kx ky ni nn");
  int Nx = atoi(argv[1]);
  int Ny = atoi(argv[2]);
  int Kx = atoi(argv[3]);
  int Ky = atoi(argv[4]);
  int Ni = atoi(argv[5]);
  int Nn = atoi(argv[6]);
  printf("Nx:%d Ny:%d Kx:%d Ky:%d Ni:%d Nn:%d\n", Nx, Ny, Kx, Ky, Ni, Nn);

  
  // SYNAPSE_SIZE can safely fit in an int for both toy example convolution sizes
  // neuron_i_size also fits in an int
  // the rest are smaller
  int neuron_i_size = NYPAD * NXPAD * Ni;
  int neuron_n_size = NYSCL * NXSCL * Nn;
  int neuron_n2_size = neuron_n_size;
  
  cout << "synapse size: " << SYNAPSE_SIZE << '\n';
  cout << "neuron i size: " << neuron_i_size << '\n';
  cout << "neuron n size: " << neuron_n_size << '\n';
  cout << "neuron n2 size: " << neuron_n2_size << '\n';
  cout << "VTYPE size: " << sizeof(VTYPE) << '\n';

  cout << "allocating memory\n";

  VTYPE* synapse, *neuron_i, *neuron_n, *neuron_n2;
  check(cudaMallocManaged((void**) &synapse, sizeof(VTYPE) * SYNAPSE_SIZE));
  check(cudaMallocManaged((void**) &neuron_i, sizeof(VTYPE) * neuron_i_size));
  check(cudaMallocManaged((void**) &neuron_n, sizeof(VTYPE) * neuron_n_size));
  check(cudaMallocManaged((void**) &neuron_n2, sizeof(VTYPE) * neuron_n2_size));

  check(cudaDeviceSynchronize());

  cout << "fill arrays\n";
  fill(synapse, SYNAPSE_SIZE);
  fill(neuron_i, neuron_i_size);


  cout << "starting computation\n";

  //Simple Version
  begin_roi();
  convolution_layer<<<1,1>>>(synapse, neuron_i, neuron_n, Nx, Ny, Kx, Ky, Ni, Nn);
  end_roi();

  cout << "simple version complete!\n";  

  cout << "starting parallel computation\n";

  //Simple Version
  begin_roi();
  convolution_layer_parallel<<<1,1>>>(synapse, neuron_i, neuron_n2, Nx, Ny, Kx, Ky, Ni, Nn);
  end_roi();

  cout << "parallel version complete!\n";  


  /*
  //Blocked Version
  begin_roi();
  convolution_layer_blocked(*synapse,*neuron_i,*neuron_n2);
  end_roi();


  cout << "blocked computation complete!\n";  

  */
  compare(neuron_n, neuron_n2,NYSCL*NXSCL*Nn);

  cout << "done\n";
}


