#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DeconvolutionLayer<Dtype>::compute_output_shape() {
  // Initialize with automatic calculations.
  this->height_out_ = this->stride_h_ * (this->height_ - 1) + this->kernel_h_
      - 2 * this->pad_h_;
  this->width_out_ = this->stride_w_ * (this->width_ - 1) + this->kernel_w_
      - 2 * this->pad_w_;
  // Check whether to hard code the output size
  if (this->layer_param_.convolution_param().has_out()) {
    CHECK(this->layer_param_.convolution_param().out() >= this->height_out_
        && this->layer_param_.convolution_param().out() - this->height_out_ < this->stride_h_)
        << "Output height should be within [height_out_, height_out + stride_h - 1]";
    CHECK(this->layer_param_.convolution_param().out() >= this->width_out_
        && this->layer_param_.convolution_param().out() - this->width_out_ < this->stride_w_)
        << "Output width should be within [width_out_, width_out + stride_w - 1]";
    this->height_out_ = this->layer_param_.convolution_param().out();
    this->width_out_= this->layer_param_.convolution_param().out();
  }
  if (this->layer_param_.convolution_param().has_out_h()
          && this->layer_param_.convolution_param().has_out_w()) {
    CHECK(this->layer_param_.convolution_param().out_h() >= this->height_out_
        && this->layer_param_.convolution_param().out_h() - this->height_out_ < this->stride_h_)
        << "Output height should be within [height_out_, height_out + stride_h - 1]";
    CHECK(this->layer_param_.convolution_param().out_w() >= this->width_out_
        && this->layer_param_.convolution_param().out_w() - this->width_out_ < this->stride_w_)
        << "Output width should be within [width_out_, width_out + stride_w - 1]";
    this->height_out_ = this->layer_param_.convolution_param().out_h();
    this->width_out_= this->layer_param_.convolution_param().out_w();
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // Gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(top_diff + top[i]->offset(n),
              bottom_data + bottom[i]->offset(n), weight_diff);
        }
        // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
        // we might have just computed above.
        if (propagate_down[i]) {
          this->forward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n),
              this->param_propagate_down_[0]);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DeconvolutionLayer);
#endif

INSTANTIATE_CLASS(DeconvolutionLayer);
REGISTER_LAYER_CLASS(Deconvolution);

}  // namespace caffe
