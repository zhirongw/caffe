#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ChnwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* mult_data = multiplier_.mutable_gpu_data();
  const Dtype* bottom_data_a = bottom[0]->gpu_data();
  const Dtype* bottom_data_b = bottom[1]->cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      for (int c = 0; c < bottom[0]->shape(1); ++c) {
        caffe_gpu_set(dim_, Dtype(bottom_data_b[c]), mult_data);
        caffe_gpu_mul(dim_, bottom_data_a, mult_data, top_data);
        bottom_data_a += dim_;
        top_data += dim_;
      }
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, Dtype(0.), top_data);
    caffe_gpu_axpy(count, coeffs_[0], bottom_data_a, top_data);
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      for (int c = 0; c < bottom[0]->shape(1); ++c) {
        caffe_gpu_set(dim_, Dtype(bottom_data_b[c]), mult_data);
        caffe_gpu_axpy(dim_, coeffs_[1], mult_data, top_data);
        bottom_data_a += dim_;
        top_data += dim_;
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown channelwise operation.";
  }
}

template <typename Dtype>
void ChnwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  Dtype* mult_data = multiplier_.mutable_gpu_data();
  switch (op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data_b = bottom[1]->cpu_data();
        Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < bottom[0]->shape(1); ++c) {
            caffe_gpu_set(dim_, Dtype(bottom_data_b[c]), mult_data);
            caffe_gpu_mul(dim_, top_diff, mult_data, bottom_diff_a);
            top_diff += dim_;
            bottom_diff_a += dim_;
          }
        }
      }
      if (propagate_down[1]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data_a = bottom[0]->gpu_data();
        Dtype* bottom_diff_b = bottom[1]->mutable_cpu_diff();
        caffe_set(bottom[0]->shape(1), Dtype(0), bottom_diff_b);
        Dtype grad;
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < bottom[0]->shape(1); ++c) {
            caffe_gpu_dot(dim_, top_diff, bottom_data_a, &grad);
            bottom_diff_b[c] += grad;
            top_diff += dim_;
            bottom_data_a += dim_;
          }
        }
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
        if (coeffs_[0] == Dtype(1)) {
          caffe_copy(count, top_diff, bottom_diff_a);
        } else {
          caffe_gpu_scale(count, coeffs_[0], top_diff, bottom_diff_a);
        }
      }
      if (propagate_down[1]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff_b = bottom[1]->mutable_cpu_diff();
        caffe_gpu_set(bottom[0]->shape(1), Dtype(0), bottom_diff_b);
        Dtype grad;
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < bottom[0]->shape(1); ++c) {
            caffe_set(dim_, Dtype(coeffs_[1]), mult_data);
            caffe_gpu_dot(dim_, top_diff, mult_data, &grad);
            bottom_diff_b[c] += grad;
            top_diff += dim_;
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown channelwise operation";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ChnwiseLayer);

}  // namespace caffe
