#include <cuda_fp16.h>
#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <algorithm>
#include <device_launch_parameters.h>

#include "caffe/common.hpp"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

///////////////////////////////////// AMAX REDUCTION ///////////////////////////////////

template<unsigned int BlockSize, typename T>
__device__ void amax_reduce_block(volatile T *sdata, T my_max, unsigned int tid) {
  volatile T* st = sdata + tid;

  tassign(st, my_max);
  __syncthreads();

  // do reduction in shared mem
  if (BlockSize >= 512) {
    if (tid < 256) {
      tmax_replace(st, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (BlockSize >= 256) {
    if (tid < 128) {
      tmax_replace(st, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (BlockSize >= 128) {
    if (tid < 64) {
      tmax_replace(st, sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32) {
    for (int i = 32; i > 0; i >>= 1) {
      tmax_replace(st, sdata[tid + i]);
    }
  }
}


// Global variable used by amax_reduce_kernel to count how many blocks have finished
__device__ unsigned int amax_blocks_count_f = 0;
__device__ unsigned int amax_blocks_count_d = 0;
__device__ unsigned int amax_blocks_count_h = 0;

template<typename T>
__device__ __inline__
unsigned int* amax_blocks_count_ptr();
template<>
__device__ __inline__
unsigned int* amax_blocks_count_ptr<float>() {
  return &amax_blocks_count_f;
}
template<>
__device__ __inline__
unsigned int* amax_blocks_count_ptr<double>() {
  return &amax_blocks_count_d;
}
template<>
__device__ __inline__
unsigned int* amax_blocks_count_ptr<__half2>() {
  return &amax_blocks_count_h;
}

template<typename T>
cudaError_t set_amax_blocks_count(unsigned int cnt);
template<>
cudaError_t set_amax_blocks_count<float>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(amax_blocks_count_f, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}
template<>
cudaError_t set_amax_blocks_count<double>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(amax_blocks_count_d, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}
template<>
cudaError_t set_amax_blocks_count<__half2>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(amax_blocks_count_h, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}

template<typename T>
__device__ __inline__
void reset_amax_blocks_count();
template<>
void reset_amax_blocks_count<float>() {
  amax_blocks_count_f = 0;
}
template<>
__device__ __inline__
void reset_amax_blocks_count<double>() {
  amax_blocks_count_d = 0;
}
template<>
__device__ __inline__
void reset_amax_blocks_count<__half2>() {
  amax_blocks_count_h = 0;
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__device__ void amax_reduce_blocks(const T *in, TR *out, unsigned int n) {
  struct __dyn_shmem__<TR> amax_shmem;
  // first level of reduction:
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
  unsigned int gridSize = BlockSize * 2 * gridDim.x;
  TR my_max = tzero<TR>();
  // We reduce multiple elements per thread. The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  while (i < n) {
    if (IsPow2 || i + BlockSize < n) {
      my_max = tmax<T, TR>(tabs(in[i]), tabs(in[i + BlockSize]));
    } else {
      tmax_replace(&my_max, tabs(in[i]));
    }
    i += gridSize;
  }
  // do reduction in shared mem
  amax_reduce_block<BlockSize>(amax_shmem.getPtr(), my_max, tid);
  // write result for this block to global mem
  if (tid == 0)
    out[blockIdx.x] = amax_shmem.getPtr()[0];
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__global__ void amax_reduce_kernel(unsigned int n, const T *in, TR *out) {
  amax_reduce_blocks<BlockSize, IsPow2>(in, out, n);

  if (gridDim.x > 1) {
    const unsigned int tid = threadIdx.x;
    struct __dyn_shmem__<TR> amax_reduce_shmem;
    __shared__ bool last_amax_block;

    // wait until all outstanding memory instructions in this thread are finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(amax_blocks_count_ptr<T>(), gridDim.x);
      last_amax_block = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    // The last block sums the results of all other blocks
    if (last_amax_block) {
      int i = tid;
      TR my_max = tzero<TR>();

      while (i < gridDim.x) {
        tmax_replace(&my_max, out[i]);
        i += BlockSize;
      }
      amax_reduce_block<BlockSize>(amax_reduce_shmem.getPtr(), my_max, tid);
      if (tid == 0) {
        out[0] = amax_reduce_shmem.getPtr()[0];
        // reset blocks count so that next run succeeds
        reset_amax_blocks_count<T>();
      }
    }
  }
}

template <typename T, typename TR>
void gpu_amax_t(const int n, const T* x, TR* result) {
  cudaStream_t stream = Caffe::thread_stream();
  const bool po2 = is_pow2(n);
  // See kernel for details
  CHECK_LE(CAFFE_CUDA_NUM_THREADS_HALF, 512);
  CHECK_GE(CAFFE_CUDA_NUM_THREADS_HALF, 128);
  const int threadsPerCta = CAFFE_CUDA_NUM_THREADS_HALF;
  const int nbrCtas = CAFFE_GET_BLOCKS_HALF(n);
  const int reductionSize = (nbrCtas + 1) * sizeof(TR);
  TR* devPtrT = reinterpret_cast<TR*>(GPUMemory::pinned_buffer(reductionSize));
  if (po2 && n > CAFFE_CUDA_NUM_THREADS_HALF) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    amax_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, true><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, devPtrT);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    amax_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, false><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, devPtrT);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  *result = devPtrT[0];
}

template <typename T>
void caffe_gpu_amax(const int n, const T* x, float* result) {
  static cudaError_t status = set_amax_blocks_count<T>(0U);  // needed just 1 time
  CUDA_CHECK(status);
  gpu_amax_t(n, x, result);
}
template
void caffe_gpu_amax<double>(const int n, const double* x, float* y);
template
void caffe_gpu_amax<float>(const int n, const float* x, float* y);
template<>
void caffe_gpu_amax<float16>(const int n, const float16* x, float* y) {
  // For odd counts we allocate extra element to speed up kernels.
  // We have to keep it clean.
  cudaStream_t stream = Caffe::thread_stream();
  if (n & 1) {
    clean_last_element(const_cast<float16*>(x) + n, stream);
  }
  const int n2 = even(n) / 2;
  static cudaError_t status = set_amax_blocks_count<__half2>(0U);  // needed just 1 time
  CUDA_CHECK(status);
  gpu_amax_t(n2, reinterpret_cast<const __half2*>(x), y);
#ifdef DEBUG
  CHECK(!isnan(*y));
  CHECK(!isinf(*y));
#endif
}



}  // namespace caffe
