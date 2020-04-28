# fp-diannao

* Karl Marrett, Eddie Huang

* This is extremely simple CUDA version of classifier/convolution dnn kernels, based on data layout and implementation from diannao paper:
http://novel.ict.ac.cn/ychen/pdf/DianNao.pdf

* Data is completely made up, but shouldn't matter for dense kernels.  Datatypes are currently set at fp32.

* Currently, Makefile is configured for example layers from VGG16.
https://arxiv.org/abs/1409.1556

## Optimization strategy
The data arrays use Cuda's modern unified memory. Different convolution or input layer sizes are specified
via runtime parameters to the executable. The simple cuda kernel example is implemented for batch size 1
is launched with a single thread and thread block and no parallel logic for simplicity.
The output of this function is therefore treated as ground truth for exploration of the other various 
strategies and compared at the end of the convolution execution.

To benchmark and profile the implementation a useful starting point is:
`NVIDIA_CUDA-10.1_Samples/1_Utilities/deviceQuery/deviceQuery`

Since this project tunes an implementation for a single processor Titan V we can list some of the
relevant parameters that are needed for a performant implementation.

The global memory is 12 GB, the data sizes we use are:

Input: Conv1: Nx=224 Ny=224 Kx=3 Ky=3 Ni=64 Nn=64
which is about: 
2^8 * 2^8 * 2^3 * 2^6 * 2^6
2^31
Since we use 2 byte floats this occupies 4 GB of global memory

After adapting the test function to verify any correctness issues with the parallelized kernel
I found that for the conv1 kernel the first run on a cold cache took 8.1 e-6 seconds
whereas the second run with the exact settings to 1.1 e-6 seconds an 8x difference. Besides
indicating that this is a serious factor that needs to be controlled when bencharking

Without substantially changing the computation patterning, the initial implementation
has direct parallelism over the output feature maps `Nn` and the height `Nx` and width `Ny` of
the input feature maps. This means the max parallelism would be `Nn * Nx * Ny` which is 
about 2^22. The peak throughput of the Titan V is 110 Teraflops or about 2^47 FLOPS
which means to take advantage of the maximum concurrency afforded by the device we should
boost the batch size which we'll treat as the hightest dimension which is also fully
parallelizable. Our mini batch size can be up to ~2^47 / 2^22 or 2^15, ~3.2e4
But global memory size

Batch size is set to 32.

Since the max number of threads per block is 1024 After reviewing the device properties

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
