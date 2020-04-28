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

// #define TEST

#define assertm(expr, msg) assert(((void) msg, expr));

// source from @talonmies
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
	  fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
   }
}
#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

void fill(VTYPE *arr, const uint64_t size) {
  for (int i=0; i < size; ++i) {
    arr[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
}

void convolution_layer(VTYPE * synapse, VTYPE* neuron_i, VTYPE* neuron_n, int Nx,
                       int Ny, int Kx, int Ky, int Ni, int Nn, int B, int Tn) {
  VTYPE sum[Nn];

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int b=0; b < B; b++) { 
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
		  // VTYPE (&synapse)[B][Ky][Kx][Nn][Ni], 
		  VTYPE sv = synapse[b * (Ky * Kx * Nn * Ni) + ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i];
		  // VTYPE (&neuron_i)[B][NYPAD][NXPAD][Ni], 
		  VTYPE nv = neuron_i[b * (NYPAD * NXPAD * Ni) + (ky + y) * NXPAD * Ni + (kx + x) * Ni + i];
		  sum[n]+=sv*nv;
		}
	  for (int n = nn; n < nn + Tn; n++) {
	    // VTYPE (&neuron_n)[B][NYSCL][NXSCL][Nn]) {
	    neuron_n[b * (NYSCL * NXSCL * Nn) + yout * (NXSCL * Nn) + xout * Nn + n] = sum[n] > 0 ? sum[n] : sum[n]/4;
	    // neuron_n[yout][xout][n] = transfer(sum[n]);
	  }
	}
	xout++; 
      }
      yout++;
    }
  }
}

__global__
void convolution_layer_gpu(VTYPE * synapse, VTYPE* neuron_i, VTYPE* neuron_n, int Nx,
                       int Ny, int Kx, int Ky, int Ni, int Nn, int B, int Tn) {
  // VTYPE sum[Nn]
  extern __shared__ VTYPE sum[];

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int b=0; b < B; b++) { 
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
		  // VTYPE (&synapse)[B][Ky][Kx][Nn][Ni], 
		  VTYPE sv = synapse[b * (Ky * Kx * Nn * Ni) + ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i];
		  // VTYPE (&neuron_i)[B][NYPAD][NXPAD][Ni], 
		  VTYPE nv = neuron_i[b * (NYPAD * NXPAD * Ni) + (ky + y) * NXPAD * Ni + (kx + x) * Ni + i];
		  sum[n]+=sv*nv;
		}
	  for (int n = nn; n < nn + Tn; n++) {
	    // VTYPE (&neuron_n)[B][NYSCL][NXSCL][Nn]) {
	    neuron_n[b * (NYSCL * NXSCL * Nn) + yout * (NXSCL * Nn) + xout * Nn + n] = sum[n] > 0 ? sum[n] : sum[n]/4;
	    // neuron_n[yout][xout][n] = transfer(sum[n]);
	  }
	}
	xout++; 
      }
      yout++;
    }
  }
}

__global__
void convolution_layer_parallel(VTYPE * synapse, VTYPE* neuron_i,
VTYPE* neuron_n, int Nx, int Ny, int Kx, int Ky, int Ni, int Nn,
int B, int tile_num_width) {
  VTYPE sum = 0;
  int b, y, x, n;
  b = blockIdx.x;
  n = blockIdx.y;
  y = blockIdx.z / tile_num_width + threadIdx.y;
  x = blockIdx.z % tile_num_width + threadIdx.x;

  // sliding window;
  for (int ky = 0; ky < Ky; ky++) {
    for (int kx = 0; kx < Kx; kx++) {
      for (int i = 0; i < Ni; i++) {
	// VTYPE (&synapse)[B][Ky][Kx][Nn][Ni]
	VTYPE sv = synapse[b * (Ky * Kx * Nn * Ni) + ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i];
	// VTYPE (&neuron_i)[B][NYPAD][NXPAD][Ni]
	VTYPE nv = neuron_i[b * (NYPAD * NXPAD * Ni) + (ky + y) * NXPAD * Ni + (kx + x) * Ni + i];
	sum +=sv*nv;
      }
    }
  }
  // VTYPE (&neuron_n)[B][NYSCL][NXSCL][Nn]) {
  neuron_n[b * (NYSCL * NXSCL * Nn) + y * (NXSCL * Nn) + x * Nn + n] = sum > 0 ? sum : sum/4;
}

__global__
void convolution_layer_shared(VTYPE * synapse,
   VTYPE* neuron_i, VTYPE* neuron_n, int Nx, 
   int Ny, int Kx, int Ky, int Ni, int Nn,
   int B, int tile_num_width, int Tn, int neuron_i_tile_width) {

  // this thread still operates over a single output
  //  neuron element, as indicated by the single sum element
  VTYPE sum = 0;
  int b, y, x, n, y_base, x_base, yoff, xoff;
  b = blockIdx.x; // idx for each batch sample
  n = blockIdx.y; // idx for each output neuron_n
  // calculate the start index for this block 
  // in terms of the output neurons (neuron_n) 
  // height and width respectively
  y_base = (blockIdx.z / tile_num_width) * Tn;
  x_base = (blockIdx.z % tile_num_width) * Tn;
  yoff = threadIdx.y;
  xoff = threadIdx.x;
  // note these values below indicate the actual location
  // of this thread with respect to the output and 
  // inputs
  y = y_base + yoff;
  x = x_base + xoff;

  extern __shared__ VTYPE shared_mem[];
  // neuron tile is (Tn + K - 1) ^2
  // to account for a full padded sweep
  VTYPE* shared_neuron_i = &shared_mem[0];

  // synapse tile is K by K and starts after neuron_i's portion above
  VTYPE* shared_synapse = &shared_mem[neuron_i_tile_width 
     * neuron_i_tile_width];

  // sliding window;
  for (int i = 0; i < Ni; i++) { // for each input neuron_i

    // bring this blocks synapses into shared memory
    if ((yoff < Ky) && (xoff < Kx)) {
      // each thread retrieves from global memory once
      // VTYPE (&synapse)[B][Ky][Kx][Nn][Ni], 
      // notice that yoff and xoff cover the full Kx * Ky 2D kernel
      shared_synapse[yoff * Kx + xoff] = 
	synapse[b * (Ky * Kx * Nn * Ni) + yoff * (Kx * Nn * Ni) + xoff * (Nn * Ni) + n * Ni + i];
    }
    // after sync all synapses for this neuron n sample
    // and input i sample are properly in shared
    __syncthreads();

   // bring input neuron_i into shared memory
   // start indexing at this blocks start location
   // for simplicity
   for (int yy=y; yy < y_base + neuron_i_tile_width; yy += Tn) {
     for (int xx=x; xx < x_base + neuron_i_tile_width; xx += Tn) {
	// VTYPE (&neuron_i)[B][NYPAD][NXPAD][Ni]
       shared_neuron_i[(yy - y_base) * neuron_i_tile_width + (xx - x_base)]
          = neuron_i[b * (NYPAD * NXPAD * Ni) + (y) * NXPAD * Ni + (x) * Ni + i];
     }
   }
   // after sync all synapses for this neuron_i batch b
   // and input i sample are properly in shared
   __syncthreads();


    for (int ky = 0; ky < Ky; ky++) {
      for (int kx = 0; kx < Kx; kx++) {
	// VTYPE (&synapse_shared)
        // [neuron_i_tile_width][neuron_i_tile_width]
	VTYPE sv = shared_synapse[ky * neuron_i_tile_width 
          + kx];	
        // VTYPE (&shared_neuron_i)
        // [neuron_i_tile_width][neuron_i_tile_width]
	VTYPE nv = shared_neuron_i[(ky + y) * 
          neuron_i_tile_width + (kx + x)];

	sum += sv*nv;
      }
    }
  }
  // VTYPE (&neuron_n)[B][NYSCL][NXSCL][Nn]) {
  neuron_n[b * (NYSCL * NXSCL * Nn) + y * (NXSCL * Nn) + x * Nn + n] = sum > 0 ? sum : sum/4;
}

int main(const int argc, const char** argv) {
  assertm(argc == 9, "8 args required usage: ./convolution nx ny kx ky ni nn batch_size tile_number");
  int Nx = atoi(argv[1]);
  int Ny = atoi(argv[2]);
  int Kx = atoi(argv[3]);
  int Ky = atoi(argv[4]);
  int Ni = atoi(argv[5]);
  int Nn = atoi(argv[6]);
  int B = atoi(argv[7]);
  int Tn = atoi(argv[8]);
  printf("Nx:%d Ny:%d Kx:%d Ky:%d Ni:%d Nn:%d\n", Nx, Ny, Kx, Ky, Ni, Nn);

  // SYNAPSE_SIZE can safely fit in an int for both toy example convolution sizes
  // neuron_i_size also fits in an int
  // the rest are smaller
  uint32_t SYNAPSE_SIZE = static_cast<uint32_t>(Ky) *Kx*Nn*Ni;
  uint32_t neuron_i_size = static_cast<uint32_t>(B) * NYPAD * NXPAD * Ni;
  uint32_t neuron_n_size = static_cast<uint32_t>(B) * NYSCL * NXSCL * Nn;
  uint32_t neuron_n2_size = static_cast<uint32_t>(B) * neuron_n_size;
  
  cout << "batch size: " << B << '\n';
  cout << "Tile size: " << Tn << '\n';
  cout << "synapse size: " << SYNAPSE_SIZE << '\n';
  cout << "neuron i size: " << neuron_i_size << '\n';
  cout << "neuron n size: " << neuron_n_size << '\n';
  cout << "neuron n2 size: " << neuron_n2_size << '\n';
  cout << "VTYPE size: " << sizeof(VTYPE) << '\n';
  assertm(Tn * Tn < 2048, "tile_size can not lead to > 2048 threads in a block");

  cout << "allocating memory\n";

  VTYPE* synapse, *neuron_i, *neuron_n, *neuron_n2;
  check(cudaMallocManaged((void**) &synapse, sizeof(VTYPE) * SYNAPSE_SIZE));
  check(cudaMallocManaged((void**) &neuron_i, sizeof(VTYPE) * neuron_i_size));
  check(cudaMallocManaged((void**) &neuron_n, sizeof(VTYPE) * neuron_n_size));
  check(cudaMallocManaged((void**) &neuron_n2, sizeof(VTYPE) * neuron_n2_size));

  check(cudaDeviceSynchronize());

  cout << "fill arrays\n";
  begin_roi();
  fill(synapse, SYNAPSE_SIZE);
  fill(neuron_i, neuron_i_size);
  end_roi();

  size_t shared_mem_size;
#ifdef TEST
#ifdef CPU
  cout << "starting sequential cpu\n";
  begin_roi();

  convolution_layer(synapse, neuron_i, neuron_n, Nx, Ny, Kx, Ky, Ni, Nn, B, Tn);

  uint64_t cpu_baseline = end_roi();
#endif

  cout << "starting sequential gpu\n";
  shared_mem_size = sizeof(VTYPE) * Nn;
  begin_roi();

  convolution_layer_gpu<<<1,1, shared_mem_size>>>(synapse, neuron_i, neuron_n, Nx, Ny, Kx, Ky, Ni, Nn, B, Tn);
  check(cudaPeekAtLastError());

  check(cudaDeviceSynchronize());

  uint64_t baseline = end_roi();
#endif

  cout << "starting parallel computation\n";

  begin_roi();
  int tile_num_width = NXSCL / Tn;
  int tile_num_height = NYSCL / Tn;
  int total_tile_num = tile_num_width * tile_num_height;
  dim3 blockDim(Tn, Tn, 1);
  dim3 gridDim(B, Nn, total_tile_num);
  printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
      gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  convolution_layer_parallel<<<gridDim, blockDim>>>(synapse, neuron_i, neuron_n2, Nx, Ny, Kx, Ky, Ni, Nn, B, tile_num_width);
  check(cudaPeekAtLastError());
  check(cudaDeviceSynchronize());
  uint64_t parallelized = end_roi();

  cout << "starting parallel shared memory computation\n";

  begin_roi();
  assertm(Kx == Ky, "Kx must equal Ky in this current implementation");
  int neuron_i_tile_width = Tn + Kx - 1;
  // this shared memory will hold both the 
  // tile of the kernel and the tile of the neuron_i
  shared_mem_size = sizeof(VTYPE) * (neuron_i_tile_width * neuron_i_tile_width + Kx * Kx);
  convolution_layer_shared<<<gridDim, blockDim, shared_mem_size>>>(synapse, neuron_i, neuron_n2, Nx, Ny, Kx, Ky, Ni, Nn, B, tile_num_width, Tn, neuron_i_tile_width);
  check(cudaPeekAtLastError());
  check(cudaDeviceSynchronize());
  uint64_t shared = end_roi();
  cout << "shared version speedup factor compared to parallelized: " << parallelized / shared << '\n';

#ifdef TEST
 cout << "parallel version speedup factor to baseline: " << baseline / parallelized << '\n';
 cout << "shared version speedup factor to baseline: " << baseline / shared << '\n';
 if (parallelized > baseline)  {
   cout << "Warning parallelized version took longer than serial\n";
 }

 // as a host function this may cause nvprof to hang silently
 compare(neuron_n, neuron_n2,NYSCL*NXSCL*Nn);
#endif


 check(cudaFree(synapse));
 check(cudaFree(neuron_i));
 check(cudaFree(neuron_n));
 check(cudaFree(neuron_n2));
 check(cudaDeviceReset());

 cout << "done\n";
}
