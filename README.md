# fp-diannao

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

### CUDNN

To improve memory further beyond using shared memory, a logical next step would be to convert the convolution
into matrix multiplication [1]. However, in order to take advantage of this to the full extent
most sources recommend using the built in GEMM operation optimized for cuda devices [2].

Doing this requires an unroll rearrangement of the neuron_i array to facilitate a matrix
multiply operation to produce each output element. Matrix multiply can further reduce pressure
on global memory accesses as it has a much higher computation to communication ratio
however it requires that the inputs are replicated `Kx * Ky` times.

[1] Kumar Chellapilla, Sidd Puri, Patrice Simard. High Performance Convolutional Neural Networks
for Document Processing. Tenth International Workshop on Frontiers in Handwriting Recognition,
Université de Rennes 1, Oct 2006, La Baule (France). ffinria-00112631f

[2] Programming Massively Parallel Processors. Kirk et al.
