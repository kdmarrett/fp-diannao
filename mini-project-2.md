For this project we model the performance of a two core kernels found ubiquitously
across the neural network space, matrix multiply (GEMM) and convolution. As a
baseline we start from a basic model as encoded via the relation in the starter
csv file. At its core these equations take in the most common parameters of
convolution and GEMM that encode the total operation that will be performed for
example height, width and channels of input, the number of samples per batch
and the number and size of filters. Note since other parameters which would
meaningfully effect both total operations (such as padding and stride) and
runtime are held constant they are ignored for the remainder of the report.

In a naive implementation of convolution where for loops explicitly define
discrete convolution as would be defined in pseudocode, a roofline analysis or
the basic equations used in this csv file would be sufficient. However, much of
the efficiency gains in highly optimized convolution libraries such as cudnn
come from algorithmic efficiency gains, in other words changes to the
computation of the algorithm such that less floating point operations are
conducted. This report discusses and models the algorithms with reported
benchmarks on the V100: winograd, winograd_nonfused and implicit_precomp_gemm
and the fft complexity. 

### Algorithms 

#### Winograd
Winograd reduces the mutliplication operation count by a factor of
2.25x as demonstrated in a comparision of various convolution algorithms
at an added cost to total addition operations which are generally less
costly to performance. The total addition account scales quadratically
with kernel size which is why cudnn only allows the winograd filtering
for small kernel sizes.

#### GEMM
The GEMM requires an unroll rearrangement of the input array to
facilitate a matrix multiply operation to produce each output element. This
unroll places all elements required by the output into a single column or row
depending on format. The filters are similarly arranged such that a row of
inputs can multiply a column of filters to produce a single output element in
GEMM fashion. Matrix multiply can further reduce pressure on global memory
accesses as it has a much higher computation to communication ratio. Note that
it requires that the inputs are replicated Kx * Ky times which can be a serious
concern on GPUs with smaller global memory sizes or larger models. At an
ecosystem level, GEMM also benefits being highly optimized for various target
hardwares since it stands at the center of many needed transformations, as
indicated for example by being part of the cuBLAS library. To be clear, 
these optimizations are at the microarchitecture level, for example by
leveraging standardized and stabilized implementation of NVidia's GPU architecture.
More relevant to our model is the fact that the algorithm itself trades
these architecture optimizations for memory footprint as unrolling
the inputs for example requires duplication for all strides of the kernel.
We should also note that the benchmarks represent the `GEMM-impl-precomp` 
which, instead of storing transformed inputs in a temporary array,
transforms on the fly during the kernel execution. This is on average
empirically the fastest of the GEMM cudnn variants (Jorda et al.).

#### CUFFT

Leveraging the frequency domain to compute convolution is a bit more
difficult to model since it involves distinct kernels and total 
floating point operation count that have only indirect relation
to the original parameters. For example, a cudnn convolution will
first conduct the fourier transform for the input and kernel,
element-wise multiply the two and take the inverse fourier transform
to place the output back in the original domain. FFT performance
itself is dependent on unique properties such as divisibility by
2 of each dimension. This is quite a high latency to introduce
but the cost of converting into and out of the frequency domain
is amortized if inputs or filters are reused in computation as
would be the case for high batch sizes. Note this is also
verified by the high 33 TOPS recorded for the cufft based convolution
on n=512.

We should not sidestep the issue that launching cuda kernels has serious
latency overhead which is why algorithms that fuse data orientatio or movement
within computation kernel tend to have generalized better performance. This is
a particularly important point for CUFFT which additionally calls two kernels
`flip_filter` `compute_gemm_pointers`. As an illustrative example, if the batch
size is too small additional kernel calls such as these exceed the runtime of
the key execution kernels.

## Precision 
The default data type used in floating point (32-bit, hence the 4B used
in the calculations). Tensor cores on the V100 enable FP16 (2B) at 
a reported 125 TFLOP peak throughput. TCs are specialized ALUs for 
FP16 but can currently only be leveraged for a subset of cudnn
algorithms such as: X.

## Discussion and Conclusion
In our analysis of performance for convolution we focus on 
kernel algorithm. The algorithm choice is foundational or 
upstream in modeling as it determines the algorithmic complexity
and thus total operation count for any given total
input and kernel size. However since cudnn has a closed source
method for selecting the fastest algorithm for given inputs
and kernels, a better model would specify the kernel parameters
and farm the selected algorithm as a front end for further 
estimating performance specific to each algorithm. Besides being
somewhat of a hack, modeling should focus on algorithms then
hardware and micrarchitecutre--not on library implementations.

References:

http://arxiv.org/abs/1509.09308
