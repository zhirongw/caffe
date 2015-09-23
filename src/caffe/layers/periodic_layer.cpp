#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PeriodicLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  PeriodicParameter periodic_param = this->layer_param().periodic_param();
  int channels = bottom[0]->channels();
  channel_shared_ = periodic_param.channel_shared();
  lw_ = this->layer_param().periodic_param().binary_loss();
  LOG(INFO) << "binary loss weight of periodic layer: " << lw_;
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > omega_filler(GetFiller<Dtype>(
        periodic_param.omega_filler()));
    omega_filler->Fill(this->blobs_[0].get());
    shared_ptr<Filler<Dtype> > phase_filler(GetFiller<Dtype>(
        periodic_param.phase_filler()));
    phase_filler->Fill(this->blobs_[1].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "omega size is inconsistent with prototxt config";
    CHECK_EQ(this->blobs_[1]->count(), 1)
        << "phase size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "omega size is inconsistent with prototxt config";
    CHECK_EQ(this->blobs_[1]->count(), channels)
        << "phase size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void PeriodicLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void PeriodicLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* omega_data = this->blobs_[0]->cpu_data();
  const Dtype* phase_data = this->blobs_[1]->cpu_data();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  const int div_factor = channel_shared_ ? channels : 1;
  switch (this->layer_param_.periodic_param().periodic_function()) {
  case PeriodicParameter_PeriodicFunction_SIN:
    // if channel_shared, channel index in the following computation becomes
    // always zero.
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      top_data[i] = std::sin(bottom_data[i] * omega_data[c] + phase_data[c]);
    }
    break;
  case PeriodicParameter_PeriodicFunction_TRI:
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      Dtype x = bottom_data[i] * omega_data[c] + phase_data[c];
      int nslope = int(std::floor(x)) % 2;
      top_data[i] = nslope ? ceil(x) - x : x - floor(x);
    }
    break;
  case PeriodicParameter_PeriodicFunction_BND:
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      top_data[i] = std::max(Dtype(-0.5), std::min(Dtype(0.5), 
          bottom_data[i] * omega_data[c])) + 0.5;
    }
    break;
  default:
    LOG(FATAL) << "Unknown Periodic Function";
  }
}

template <typename Dtype>
void PeriodicLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* omega_data = this->blobs_[0]->cpu_data();
  const Dtype* phase_data = this->blobs_[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const int cdim = channels * dim;

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  switch(this->layer_param_.periodic_param().periodic_function()) {
  case PeriodicParameter_PeriodicFunction_SIN:
    // Propagte to param
    // Since to write bottom diff will affect top diff if top and bottom blobs
    // are identical (in-place computaion), we first compute param backward to
    // keep top_diff unchanged.
    for (int n = 0; n < bottom[0]->num(); ++n) {
      Dtype* backward_buffer_diff = backward_buff_.mutable_cpu_diff();
      for (int i = 0; i < cdim; ++i) {
        int c = (i / dim) % channels / div_factor;
        backward_buffer_diff[i] = top_diff[i] *
            std::cos(bottom_data[i] * omega_data[c] + phase_data[c]);
      }

      // Propagate to phase data
      if (this->param_propagate_down_[1]) {
        Dtype* phase_diff = this->blobs_[1]->mutable_cpu_diff();
        if (channel_shared_) {
          Dtype d;
          caffe_cpu_dot<Dtype>(cdim, backward_buff_.cpu_diff(),
              multiplier_.cpu_data());
          caffe_add_scalar(this->blobs_[0]->count(), Dtype(d), phase_diff);
        } else {
          caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
              backward_buff_.cpu_diff(), multiplier_.cpu_data(), 1.,
              phase_diff);
        }
      }

      // Propagate to bottom
      if (propagate_down[0]) {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        bottom_diff = bottom_diff + bottom[0]->offset(n);
        const Dtype* backward_buffer_diff = backward_buff_.cpu_diff();
        for (int i = 0; i < cdim; ++i) {
          int c = (i / dim) % channels / div_factor;
          bottom_diff[i] = omega_data[c] * backward_buffer_diff[i];
        }
      }

      // Propagate to omega
      if (this->param_propagate_down_[0]) {
        Dtype* omega_diff = this->blobs_[0]->mutable_cpu_diff();
        caffe_mul(cdim, backward_buff_.cpu_diff(),
          bottom_data, backward_buff_.mutable_cpu_diff());
        if (channel_shared_) {
          Dtype d;
          d = caffe_cpu_dot<Dtype>(cdim, backward_buff_.cpu_diff(),
              multiplier_.cpu_data());
          caffe_add_scalar(this->blobs_[0]->count(), Dtype(d), omega_diff);
        } else {
          caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
              backward_buff_.cpu_diff(), multiplier_.cpu_data(), 1.,
              omega_diff);
        }
      }
      bottom_data += cdim;
      top_diff += cdim;
    }
    break;
  case PeriodicParameter_PeriodicFunction_TRI:
    // Propagte to param
    // Since to write bottom diff will affect top diff if top and bottom blobs
    // are identical (in-place computaion), we first compute param backward to
    // keep top_diff unchanged.
    for (int n = 0; n < bottom[0]->num(); ++n) {
      Dtype* backward_buffer_diff = backward_buff_.mutable_cpu_diff();
      for (int i = 0; i < cdim; ++i) {
        int c = (i / dim) % channels / div_factor;
        Dtype x = bottom_data[i] * omega_data[c] + phase_data[c];
        int nslope = int(floor(x)) % 2;
        backward_buffer_diff[i] = nslope ? - top_diff[i] : top_diff[i];
      }

      // Propagate to phase data
      if (this->param_propagate_down_[1]) {
        Dtype* phase_diff = this->blobs_[1]->mutable_cpu_diff();
        if (channel_shared_) {
          Dtype d;
          caffe_cpu_dot<Dtype>(cdim, backward_buff_.cpu_diff(),
              multiplier_.cpu_data());
          caffe_add_scalar(this->blobs_[0]->count(), Dtype(d), phase_diff);
        } else {
          caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
              backward_buff_.cpu_diff(), multiplier_.cpu_data(), 1.,
              phase_diff);
        }
      }

      // Propagate to bottom
      if (propagate_down[0]) {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        bottom_diff = bottom_diff + bottom[0]->offset(n);
        const Dtype* backward_buffer_diff = backward_buff_.cpu_diff();
        for (int i = 0; i < cdim; ++i) {
          int c = (i / dim) % channels / div_factor;
          bottom_diff[i] = omega_data[c] * backward_buffer_diff[i];
        }
      }

      // Propagate to omega
      if (this->param_propagate_down_[0]) {
        Dtype* omega_diff = this->blobs_[0]->mutable_cpu_diff();
        caffe_mul(cdim, backward_buff_.cpu_diff(),
          bottom_data, backward_buff_.mutable_cpu_diff());
        if (channel_shared_) {
          Dtype d;
          d = caffe_cpu_dot<Dtype>(cdim, backward_buff_.cpu_diff(),
              multiplier_.cpu_data());
          caffe_add_scalar(this->blobs_[0]->count(), Dtype(d), omega_diff);
        } else {
          caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
              backward_buff_.cpu_diff(), multiplier_.cpu_data(), 1.,
              omega_diff);
        }
      }
      bottom_data += cdim;
      top_diff += cdim;
    }
    break;
  case PeriodicParameter_PeriodicFunction_BND:
    {
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* omega_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < top[0]->count(); ++i) {
      int c = (i / dim) % channels / div_factor;
      if (this->param_propagate_down_[0]) {
        omega_diff[c] += (top_diff[i] + lw_ * std::log((1 - top_data[i]) / top_data[i]) / std::log(2.0)) 
            * bottom_data[i] * (top_data[i] < 1) * (top_data[i] > 0);
      }
      if (propagate_down[0]) {
        bottom_diff[i] = (top_diff[i] + lw_ * (top_data[i] < 0.5 ? 1 : -1))
            * omega_data[c] * (top_data[i] < 1) * (top_data[i] > 0);
      }
    }
    }
    break;
  default:
    LOG(FATAL) << "Unknown Periodic Function";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PeriodicLayer);
#endif

INSTANTIATE_CLASS(PeriodicLayer);
REGISTER_LAYER_CLASS(Periodic);

}  // namespace caffe
