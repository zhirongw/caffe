#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ChnwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Chnwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "Chnwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  axis_ = 2;
  num_ = bottom[0]->count(0, axis_);
  dim_ = bottom[0]->count(axis_);
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
  CHECK(!stable_prod_grad_ || (bottom[0] != top[0]))
        << "Only support unstable production gradient for in-place calculation";
}

template <typename Dtype>
void ChnwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape(1) == bottom[0]->shape(1));
  }
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_shape(1, dim_);
  multiplier_.Reshape(mult_shape);
}

template <typename Dtype>
void ChnwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_a = bottom[0]->cpu_data();
  const Dtype* bottom_data_b = bottom[1]->cpu_data();
  Dtype* mult_data = multiplier_.mutable_cpu_data();
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      for (int c = 0; c < bottom[0]->shape(1); ++c) {
        caffe_set(dim_, Dtype(bottom_data_b[c]), mult_data);
        caffe_mul(dim_, bottom_data_a, mult_data, top_data);
        bottom_data_a += dim_;
        top_data += dim_;
      }
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_set(count, Dtype(0), top_data);
    caffe_axpy(count, coeffs_[0], bottom_data_a, top_data);
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      for (int c = 0; c < bottom[0]->shape(1); ++c) {
        caffe_set(dim_, Dtype(bottom_data_b[c]), mult_data);
        caffe_axpy(dim_, coeffs_[1], mult_data, top_data);
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
void ChnwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  Dtype* mult_data = multiplier_.mutable_cpu_data();
  switch (op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      if (propagate_down[1]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_data_a = bottom[0]->mutable_cpu_data();
        const Dtype* bottom_data_b = bottom[1]->cpu_data();
        const Dtype* top_data = top[0]->cpu_data();
        Dtype* bottom_diff_b = bottom[1]->mutable_cpu_diff();
        caffe_set(bottom[0]->shape(1), Dtype(0), bottom_diff_b);
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < bottom[0]->shape(1); ++c) {
            if (!stable_prod_grad_) {
              caffe_set(dim_, bottom_data_b[c], mult_data);
              caffe_div(dim_, top_data, mult_data, bottom_data_a);
              top_data += dim_;
            }
            bottom_diff_b[c] += caffe_cpu_dot(dim_, top_diff, bottom_data_a);
            bottom_data_a += dim_;
            top_diff += dim_;
          }
        }
      }
      if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data_b = bottom[1]->cpu_data();
        Dtype* bottom_diff_a = bottom[0]->mutable_cpu_diff();
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < bottom[0]->shape(1); ++c) {
            caffe_set(dim_, bottom_data_b[c], mult_data);
            caffe_mul(dim_, top_diff, mult_data, bottom_diff_a);
            bottom_diff_a += dim_;
            top_diff += dim_;
          }
        }
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      if (propagate_down[1]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff_b = bottom[1]->mutable_cpu_diff();
        caffe_set(bottom[0]->shape(1), Dtype(0), bottom_diff_b);
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < bottom[0]->shape(1); ++c) {
            caffe_set(dim_, Dtype(coeffs_[1]), mult_data);
            bottom_diff_b[c] += caffe_cpu_dot(dim_, top_diff, mult_data);
            top_diff += dim_;
          }
        }
      }
      if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff_a = bottom[0]->mutable_cpu_diff();
        if (coeffs_[0] == Dtype(1)) {
          caffe_copy(count, top_diff, bottom_diff_a);
        } else {
          caffe_cpu_scale(count, coeffs_[0], top_diff, bottom_diff_a);
        }
      }
      break;
    default:
        LOG(FATAL) << "Unknown channelwise operation.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(ChnwiseLayer);
#endif

INSTANTIATE_CLASS(ChnwiseLayer);
REGISTER_LAYER_CLASS(Chnwise);

}  // namespace caffe
