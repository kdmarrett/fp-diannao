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

