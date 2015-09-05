#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// CUDA kernel for sin forward
template <typename Dtype>
__global__ void SinForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* omega_data,
    const Dtype* phase_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = sin(in[index] * omega_data[c] + phase_data[c]);
  }
}

// CUDA kernel for triangle forward
template <typename Dtype>
__global__ void TriForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* omega_data,
    const Dtype* phase_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    Dtype x = in[index] * omega_data[c] + phase_data[c];
    int nslope = int(floorf(x)) % 2;
    out[index] = nslope ? ceilf(x) - x : x - floorf(x);
  }
}

// CUDA kernel for sin backward
template <typename Dtype>
__global__ void SinBackward(const int n, const int channels,
    const int dim, const Dtype* in_diff, const Dtype* in_data,
    const Dtype* omega_data, const Dtype* phase_data, Dtype* out_diff,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index]
        * cos(in_data[index] * omega_data[c] + phase_data[c]);
  }
}

// CUDA kernel for sin backward
template <typename Dtype>
__global__ void TriBackward(const int n, const int channels,
    const int dim, const Dtype* in_diff, const Dtype* in_data,
    const Dtype* omega_data, const Dtype* phase_data, Dtype* out_diff,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    Dtype x = in_data[index] * omega_data[c] + phase_data[c];
    int nslope = int(floorf(x)) % 2;
    out_diff[index] = nslope ? - in_diff[index] : in_diff[index];
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void BottomBackward(const int n, const int channels,
    const int dim, const Dtype* in_diff, Dtype* out_diff,
    const Dtype* omega_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * omega_data[c];
  }
}

template <typename Dtype>
void PeriodicLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* omega_data = this->blobs_[0]->gpu_data();
  const Dtype* phase_data = this->blobs_[1]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  switch(this->layer_param_.periodic_param().periodic_function()) {
  case PeriodicParameter_PeriodicFunction_SIN:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SinForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, bottom_data, top_data, omega_data,
        phase_data, div_factor);
    CUDA_POST_KERNEL_CHECK;
    break;
  case PeriodicParameter_PeriodicFunction_TRI:
    // NOLINT_NEXT_LINE(whitespace/operators)
    TriForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, bottom_data, top_data, omega_data,
        phase_data, div_factor);
    CUDA_POST_KERNEL_CHECK;
    break;
  default:
    LOG(FATAL) << "Unknown Periodic Function";
  }
}

template <typename Dtype>
void PeriodicLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* omega_data = this->blobs_[0]->gpu_data();
  const Dtype* phase_data = this->blobs_[1]->gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  int cdim = channels * dim;
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  switch(this->layer_param_.periodic_param().periodic_function()) {
  case PeriodicParameter_PeriodicFunction_SIN:
    for (int n = 0; n < bottom[0]->num(); ++n) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SinBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
          CAFFE_CUDA_NUM_THREADS>>>(
          cdim, channels, dim, top_diff + top[0]->offset(n),
          bottom_data + bottom[0]->offset(n), omega_data,
          phase_data, backward_buff_.mutable_gpu_diff(),
          div_factor);
      CUDA_POST_KERNEL_CHECK;

      // Propagate to phase
      if (this->param_propagate_down_[1]) {
        Dtype* phase_diff = this->blobs_[1]->mutable_gpu_diff();
        if (channel_shared_) {
          Dtype d;
          caffe_gpu_dot<Dtype>(cdim, backward_buff_.gpu_diff(),
              multiplier_.gpu_data(), &d);
          caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(d), phase_diff);
        } else {
          caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
              backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
              phase_diff);
        }
      }

      // Propagate to bottom
      if (propagate_down[0]) {
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        BottomBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            cdim, channels, dim, backward_buff_.gpu_diff(),
            bottom_diff + bottom[0]->offset(n), omega_data, div_factor);
        CUDA_POST_KERNEL_CHECK;
      }

      // Propagate to omega
      if (this->param_propagate_down_[0]) {
        Dtype* omega_diff = this->blobs_[0]->mutable_gpu_diff();
        caffe_gpu_mul(cdim, backward_buff_.gpu_diff(), bottom_data
            + bottom[0]->offset(n), backward_buff_.mutable_gpu_diff());
        if (channel_shared_) {
          Dtype d;
          caffe_gpu_dot<Dtype>(cdim, backward_buff_.gpu_diff(),
              bottom_data + bottom[0]->offset(n), &d);
          caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(d), omega_diff);
        } else {
          caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
              backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
              omega_diff);
        }
      }
    }
    break;
  case PeriodicParameter_PeriodicFunction_TRI:
    for (int n = 0; n < bottom[0]->num(); ++n) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      TriBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
          CAFFE_CUDA_NUM_THREADS>>>(
          cdim, channels, dim, top_diff + top[0]->offset(n),
          bottom_data + bottom[0]->offset(n), omega_data,
          phase_data, backward_buff_.mutable_gpu_diff(),
          div_factor);
      CUDA_POST_KERNEL_CHECK;

      // Propagate to phase
      if (this->param_propagate_down_[1]) {
        Dtype* phase_diff = this->blobs_[1]->mutable_gpu_diff();
        if (channel_shared_) {
          Dtype d;
          caffe_gpu_dot<Dtype>(cdim, backward_buff_.gpu_diff(),
              multiplier_.gpu_data(), &d);
          caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(d), phase_diff);
        } else {
          caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
              backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
              phase_diff);
        }
      }

      // Propagate to bottom
      if (propagate_down[0]) {
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        BottomBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            cdim, channels, dim, backward_buff_.gpu_diff(),
            bottom_diff + bottom[0]->offset(n), omega_data, div_factor);
        CUDA_POST_KERNEL_CHECK;
      }

      // Propagate to omega
      if (this->param_propagate_down_[0]) {
        Dtype* omega_diff = this->blobs_[0]->mutable_gpu_diff();
        caffe_gpu_mul(cdim, backward_buff_.gpu_diff(), bottom_data
            + bottom[0]->offset(n), backward_buff_.mutable_gpu_diff());
        if (channel_shared_) {
          Dtype d;
          caffe_gpu_dot<Dtype>(cdim, backward_buff_.gpu_diff(),
              bottom_data + bottom[0]->offset(n), &d);
          caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(d), omega_diff);
        } else {
          caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
              backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
              omega_diff);
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown Periodic Function";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PeriodicLayer);


}  // namespace caffe
