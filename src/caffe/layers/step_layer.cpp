#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StepLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void StepLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < count; ++i) {
    top_data[i] = Dtype(bottom_data[i] > 0);
  }
}

template <typename Dtype>
void StepLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(StepLayer);
#endif

INSTANTIATE_CLASS(StepLayer);
REGISTER_LAYER_CLASS(Step);

}  // namespace caffe
