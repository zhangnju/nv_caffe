#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void BRForward(const int count, const int inner_dim, const Dtype* in,
                          const Dtype* permut, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
}

template <typename Ftype, typename Btype>
void BatchReindexLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
                                           const vector<Blob*>& top) {
  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data<Ftype>());
  if (top[0]->count() == 0) {
    return;
  }
  int threads = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BRForward<Ftype><<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS, 0,
      Caffe::thread_stream()>>>(
      top[0]->count(), bottom[0]->count() / bottom[0]->shape(0),
      bottom[0]->gpu_data<Ftype>(), bottom[1]->gpu_data<Ftype>(),
      top[0]->mutable_gpu_data<Ftype>());
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void BRBackward(const int count, const int inner_dim,
                           const Dtype* in, const Dtype* top_indexes,
                           const Dtype* begins, const Dtype* counts,
                           Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / (inner_dim);
    out[index] = 0;
    int lower = static_cast<int>(begins[n]);
    int upper = lower + static_cast<int>(counts[n]);
    for (int i = lower; i < upper; ++i) {
      int in_n = static_cast<int>(top_indexes[i]);
      out[index] += in[in_n * (inner_dim) + index % (inner_dim)];
    }
  }
}

template <typename Ftype, typename Btype>
void BatchReindexLayer<Ftype, Btype>::Backward_gpu(
    const vector<Blob*>& top, const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backprop to index.";
  if (!propagate_down[0]) {
    return;
  }

  vector<std::pair<int, int> > mapping;
  const Btype* perm = bottom[1]->cpu_data<Btype>();
  for (int i = 0; i < bottom[1]->count(); ++i) {
    mapping.push_back(pair<int, int>(static_cast<int>(perm[i]), i));
  }
  std::sort(mapping.begin(), mapping.end(), pair_sort_first());

  // Each element of the bottom diff is potentially the sum of many top diffs.
  // However, we'd like each CUDA thread to handle exactly one output.  Hence,
  // we first pre-compute a list of lists of indices that need to be summed for
  // each output. `top_indexes` holds the data of this list of lists.  The
  // k'th element of `begins` points to the location in `top_indexes` where the
  // list for the k'th example begin, and the k'th element of `counts` is the
  // length of that list.
  vector<int> shape;
  shape.push_back(bottom[1]->count());
  TBlob<Btype> top_indexes(shape);
  shape[0] = bottom[0]->shape(0);
  TBlob<Btype> counts(shape);
  TBlob<Btype> begins(shape);
  Btype* t_i_data = top_indexes.mutable_cpu_data();
  Btype* c_data = counts.mutable_cpu_data();
  Btype* b_data = begins.mutable_cpu_data();
  caffe_set(begins.count(), Btype(-1), b_data);
  caffe_set(counts.count(), Btype(0), c_data);
  for (int i = 0; i < mapping.size(); ++i) {
    t_i_data[i] = mapping[i].second;
    if (b_data[mapping[i].first] == -1) {
      b_data[mapping[i].first] = i;
    }
    c_data[mapping[i].first] += 1;
  }

  int threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BRBackward<Btype><<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS, 0,
      Caffe::thread_stream()>>>(
      bottom[0]->count(), bottom[0]->count() / bottom[0]->shape(0),
      top[0]->gpu_diff<Btype>(), top_indexes.gpu_data(), begins.gpu_data(),
      counts.gpu_data(), bottom[0]->mutable_gpu_diff<Btype>());
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(BatchReindexLayer);

}  // namespace caffe
