# fp-diannao

* Karl Marrett, Eddie Huang

* forked cpu project, adapted for cuda, tuned for Titan V

* This is extremely simple CUDA version of classifier/convolution dnn kernels, based on data layout and implementation from diannao paper:
http://novel.ict.ac.cn/ychen/pdf/DianNao.pdf

* Data is completely made up, but shouldn't matter for dense kernels.  Datatypes are currently set at fp32.

* Currently, Makefile is configured for example layers from VGG16.
https://arxiv.org/abs/1409.1556

## Convolution

The data arrays use Cuda's modern unified memory. Different convolution or
input layer sizes are specified via runtime parameters to the executable. The
simple cuda kernel example is implemented for batch size 1 is launched with a
single thread and thread block and no parallel logic for simplicity. Due to
it's slowness it was only used for ground truth correctness along with the
ported cpu version of convolution.  Basic threading schemes were at least 2800x
faster than the single thread version understandably.

The output of this function is therefore treated as ground truth for
exploration of the other various strategies and compared at the end of the
convolution execution when the TEST macro is on.

To benchmark and profile the implementation a useful starting point is:
`NVIDIA_CUDA-10.1_Samples/1_Utilities/deviceQuery/deviceQuery`

Since this project tunes an implementation for a single processor Titan V we can list some of the
relevant parameters that are needed for a performant implementation.

The global memory is 12 GB, the data sizes we use before batching are:

Input: Conv1: Nx=224 Ny=224 Kx=3 Ky=3 Ni=64 Nn=64 B=1
which is about: 
2^8 * 2^8 * 2^3 * 2^6 * 2^6 or about 2^31
Since we use 2 byte floats this occupies 4 GB of global memory. The filter bank (synapses)
had a size of about 2^8 * 2^6 * 2^6 which is about 2^22 or 8 MB. The set of output
neurons has a size of 2^6 * 2^8 * 2^8 or 2^22 aka 8 MB. Thus we checked batch sizes
of 1 2 3 4 8 and 12. Despite batch size of 3 overriding the global memory limits on
the GPU, only a batch size of about 16 or higher caused the cuda helper checks of the
memory malloc to fail. As a performance measure we take the latency of the operation
by forcing a device synchronize host code and dividing by the total computations
of the naive implementations (product of all original for loop sizes).

Without substantially changing the computation patterning, the initial implementation
has direct parallelism over the output feature maps `Nn` and the height `Nx` and width `Ny` of
the input feature maps. This means the max parallelism would be `Nn * Nx * Ny` which is 
about 2^22. The peak throughput of the Titan V is 110 Teraflops or about 2^47 FLOPS
which means to take advantage of the maximum concurrency afforded by the device we should
boost the batch size which we'll treat as the hightest dimension which is also fully
parallelizable. We profiled across batch sizes and tile sizes in a benchmark matrix 
pattern as shown by `script.sh`. The highest performance . These implementations
rely solely on threads accessing global memory as opposed to leveraging the shared
or constant memory banks, which greatly reduces there throughput. To combat this
we implemented a shared memory approach where threads collaboratively load portions
of neuron_i and synapses needed to compute one element of the output feature maps
`neuron_n` for a single batch sample. This function could not pass the tests by 
assignment deadline so we moved on. The implementation is based off a suggested
approach in [1] however, our implementation did not pass tests so it's performance
is not reported here.

For the non-shared memory kernels shown the highest performance recorded is 82.4 GFLOPS
for a batch size of 2 and tile size of 8.

## Classification

To parallelize the classifier for the first problem size, we parallelized using on global 
work-groups on Nn and on local workgroups on Ni, with each local workgroup responsible for 
accumulating 49 elements of Ni. While a larger parallelization factor was certainly
possible, (we tried parallelizing such that each work-item only did one multiply) which led
to performance of about 25 TFlops, without reduction, we didn't have time to implement anything
beyond a lazy reduction, which netted us about 7 TFLops of performance. No batching was
applied here, though it almost certainly would have resulted in better throughput, due
to the fact that resources weren't fully utilized.

A similar strategy was applied for the second problem size, and as the Ni dimension was
a power of 2, the lazy reduction worked. However, due to a lack of batching, and 
resource utilization that was not close to full, the performance was only about 1 TFlop.

### CUDNN

Since cudnn benchmarks were provided, we elected not to benchmark it ourselves. Since cudnn 
is close source we instead elected to find what information was available regarding its implementation
and understand it's high performance. Our reasonings are detailed as follows.
To improve memory further beyond using shared memory, a logical next step would be to convert the convolution
into matrix multiplication [1]. However, in order to take advantage of this to the full extent
sources recommend using the built in GEMM operation optimized for cuda devices [2].

Doing this requires an unroll rearrangement of the neuron_i array to facilitate a matrix
multiply operation to produce each output element. Matrix multiply can further reduce pressure
on global memory accesses as it has a much higher computation to communication ratio
however it requires that the inputs are replicated `Kx * Ky` times which can be a serious concern
on GPUs with smaller global memory sizes or larger models. At an ecosystem level, 
GEMM also benefits of being highly optimized for various target hardwares since it stands at the center of many needed
transformations, as indicated for example by being part of the cuBLAS library. 

GEMM was not targeted in this report since it relies significantly on pre-optimized
function calls as opposed to working with the fundamental computation and iteratively incorporating
and learning more of the GPUs hardware features. Discussing GEMM is also important for understanding
the standard cudnn implementation of convolution. CUDNN has other implementations such as 
CUFFT--decomposing to frequency domain and multiplying--and more modern techniques such as Winograd and separable
filters. While CUDNN is closed source, various representatives from Nvidia or affiliated have
explained the basic scheme of the the GEMM-based cudnn convolution. It's major advantage to the
the simply unrolling the input feature map and leveraging cuBLAS GEMM is that it supports a streaming
approach of loading the input feature map into device memory. This is particularly useful since
the input feature map has to be duplicated and rearranged (unrolled) which is much faster in
device memory. Additionally, the streaming (or blocked) implementation automatically handles the
overlap of communication and computation, meaning a kernel is launched while the next block
of input feature map is loaded to device memory.

By not streaming portions of memory or overlapping the computation with the loads in our own convolution 
we severly limited the final bandwidth we could achieve. This causes taking all of latency
of moving the input and kernel arrays to global memory and back upfront.

#### References

[1] Kumar Chellapilla, Sidd Puri, Patrice Simard. High Performance Convolutional Neural Networks
for Document Processing. Tenth International Workshop on Frontiers in Handwriting Recognition,
Université de Rennes 1, Oct 2006, La Baule (France). ffinria-00112631f

[2] Programming Massively Parallel Processors. Kirk et al.

## Appendix

```
// classifier.cu
// Author: Eddie Huang
// MIT License

#include <iostream>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 128  // Number of Output Layers
  #define Ni 224  // Number of Input  Layers
#endif

#define gridSize 512
#define blockSize 49

#define num_ops Nn*Ni*2

#ifndef Tii
  // Tiling Sizes
  #define Tnn 512  
  #define Tii 512
  //#define Tn 5
  //#define Ti 25
  #define Tn 8
  #define Ti 49
#endif

//Arrays:
VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Ni] __attribute__((aligned(64)));
VTYPE neuron_temp[Nn][512] __attribute__((aligned(64)));
VTYPE neuron_n[Nn] __attribute__((aligned(64))),    neuron_n2[Nn] __attribute__((aligned(64)));

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
    VTYPE (&neuron_n)[Nn],   VTYPE (&neuron_temp)[Nn][512], VTYPE (&neuron_n2)[Nn]) {
  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      synapse[n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
    for(int i = 0; i < 512; ++i) {
      neuron_temp[n][i] = 0;
    }
  }
  for(int i = 0; i < Ni; ++i) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for(int n = 0; n < Nn; ++n) {
    neuron_n[n] = 0; //i;
    neuron_n2[n] = 0; //i;
  }
}

__global__ void classifier_layer(VTYPE* synapse, size_t pitch, VTYPE* neuron_i, VTYPE* neuron_n) {
  VTYPE temp=0;
  VTYPE *row = (VTYPE *)((char*)synapse + blockIdx.x*pitch);
  __shared__ VTYPE sum[512];
  for (int i = threadIdx.x; i < threadIdx.x + 49; i++) {
    temp += row[i] * neuron_i[i];
  }  
  sum[threadIdx.x] = temp;
  __syncthreads();
  for(int i=256; i>0; i/=2) {
    if(threadIdx.x<i)
      sum[threadIdx.x] += sum[threadIdx.x+i];
    __syncthreads();
  }
  if(threadIdx.x == 0) 
    neuron_n[blockIdx.x] = (sum[0]>0) ? sum[0] : sum[0]/4;
}

__global__ void classifier_layer_blocked(VTYPE* synapse, size_t pitch, VTYPE* neuron_i, 
                              VTYPE* neuron_temp) {
  int total_calc=0;
  __shared__ VTYPE sum[Tn];
  sum[threadIdx.x] = 0;
  __syncthreads();
  int nnn = blockIdx.x * 64;
  int iii = blockIdx.y * 64;
  int nn = nnn + threadIdx.x;
  int ii = iii + threadIdx.y;
  for (int n = nn; n < nn + 8; n++) {
    VTYPE sum_sc=0;
    VTYPE *row = (VTYPE *)((char*)synapse + n*pitch);
    for (int i = ii; i < ii + 8; i++) {
      sum_sc += (row[i] * neuron_i[i]);
    }
    sum[n]+=sum_sc;
    __syncthreads();
  }
  /*for (int nn = nnn; nn < nnn + Tnn; nn++) {
    neuron_n[nn] = (nn[i]>0) ? temp : temp/4;
  }*/
}

int main(int argc, char** argv) {
  cout << "initializing arrays\n";

  fill_classifier(synapse,neuron_i,neuron_n,neuron_temp,neuron_n2);

  VTYPE *d_synapse = NULL;
  size_t pitch;
  cudaMallocPitch((void **)&d_synapse, &pitch, Ni*sizeof(float), Nn);

  VTYPE *d_neuron_i = NULL;
  cudaMalloc((void **)&d_neuron_i, Nn*sizeof(float));

  VTYPE *d_neuron_n = NULL;
  cudaMalloc((void **)&d_neuron_n, Nn*sizeof(float));

  VTYPE *d_neuron_temp = NULL;
  size_t pitch2;
  cudaMallocPitch((void **)&d_synapse, &pitch2, 512*sizeof(float), Nn);

  VTYPE *d_neuron_n2 = NULL;
  cudaMalloc((void **)&d_neuron_n2, Nn*sizeof(float));

  cudaMemcpy2D(d_synapse, pitch, synapse, Ni*sizeof(float), Ni*sizeof(float), Nn, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_neuron_temp, pitch, synapse, 512*sizeof(float), 512*sizeof(float), Nn, cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron_i, neuron_i, Nn*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron_n, neuron_n, Ni*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron_n2, neuron_n2, Ni*sizeof(float), cudaMemcpyHostToDevice);

  cout << "starting computation\n";

  begin_roi();
  classifier_layer<<<4096, 512>>>(d_synapse,pitch,d_neuron_i,d_neuron_n);
  end_roi(num_ops);
 
  cout << "simple version complete!\n";  
  
/*  dim3 grids(Nn/Tnn, Ni/Tii); // 8, 49
  dim3 blocks(64, 64); 

  begin_roi();
  classifier_layer_blocked<<<grids, blocks>>>(d_synapse,pitch,d_neuron_i,d_neuron_temp);  
  end_roi(num_ops);

  cout << "blocked computation complete!\n";  */

  compare(neuron_n,neuron_n2,Nn);

  cout << "done\n";
}

```

```
// convolution.cu
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
// #define CONV_SHARED

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

  uint64_t total_flops = static_cast<uint64_t>(Kx) * Ky * Nn * Ni * Ny * Nx * B;

  cout << "fill arrays\n";
  fill(synapse, SYNAPSE_SIZE);
  fill(neuron_i, neuron_i_size);

  size_t shared_mem_size;

#ifdef TEST
#ifdef CPU
  cout << "starting sequential cpu\n";
  begin_roi();

  convolution_layer(synapse, neuron_i, neuron_n, Nx, Ny, Kx, Ky, Ni, Nn, B, Tn);

  uint64_t cpu_baseline = end_roi("sequential", total_flops);
#endif

  cout << "starting sequential gpu\n";
  shared_mem_size = sizeof(VTYPE) * Nn;
  begin_roi();

  convolution_layer_gpu<<<1,1, shared_mem_size>>>(synapse, neuron_i, neuron_n, Nx, Ny, Kx, Ky, Ni, Nn, B, Tn);
  check(cudaPeekAtLastError());

  check(cudaDeviceSynchronize());

  uint64_t baseline = end_roi("sequential gpu", total_flops);
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
  uint64_t parallelized = end_roi("parallel gpu", total_flops);

#ifdef CONV_SHARED

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
  uint64_t shared = end_roi("shared", total_flops);
  cout << "shared version speedup factor compared to parallelized: " << parallelized / shared << '\n';
#endif

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
```
