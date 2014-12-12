// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;


namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number of instances";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data and label should have the same height";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data and label should have the same width";
  // Top will contain:
  // top[0] = Sensitivity or Recall (TP/P),
  // top[1] = Specificity (TN/N),
  // top[2] = Harmonic Mean of Sens and Spec, (2/(P/TP+N/TN))
  // top[3] = Precision (TP / (TP + FP))
  // top[4] = F1 Score (2 TP / (2 TP + FP + FN))
  

  // Modified by Ziwei at August 15, 2014
  // (*top)[0]->Reshape(1, 5, 1, 1);

  int num_labels = bottom[0]->channels();
  (*top)[0]->Reshape(1, num_labels, 1, 1);
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;

  // Modified by Ziwei at August 15, 2014
  int num_labels = bottom[0]->channels();

  vector<float> count_correct(num_labels);
  count_correct.clear();

  for (int ind = 0; ind < num_labels; ++ind) {
      count_correct[ind] = 0;
  }

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();

  for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]);
    if (label > 0) {
    // Update Positive accuracy and count
      true_positive += (bottom_data[ind] >= 0);
      false_negative += (bottom_data[ind] < 0);
      count_pos++;

      // Modified by Ziwei at August 15, 2014
      count_correct[ind % num_labels] += (bottom_data[ind] > 0);
    }
    if (label < 0) {
    // Update Negative accuracy and count
      true_negative += (bottom_data[ind] < 0);
      false_positive += (bottom_data[ind] >= 0);
      count_neg++;

      // Modified by Ziwei at August 15, 2014
      count_correct[ind % num_labels] += (bottom_data[ind] < 0);
    }
  }

  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  Dtype harmmean = ((count_pos + count_neg) > 0)?
    2 / (count_pos / true_positive + count_neg / true_negative) : 0;
  Dtype precission = (true_positive > 0)?
    (true_positive / (true_positive + false_positive)) : 0;
  Dtype f1_score = (true_positive > 0)?
    2 * true_positive /
    (2 * true_positive + false_positive + false_negative) : 0;

  DLOG(INFO) << "Sensitivity: " << sensitivity;
  DLOG(INFO) << "Specificity: " << specificity;
  DLOG(INFO) << "Harmonic Mean of Sens and Spec: " << harmmean;
  DLOG(INFO) << "Precission: " << precission;
  DLOG(INFO) << "F1 Score: " << f1_score;
  
  // Modified by Ziwei at August 15, 2014
  // (*top)[0]->mutable_cpu_data()[0] = sensitivity;
  // (*top)[0]->mutable_cpu_data()[1] = specificity;
  // (*top)[0]->mutable_cpu_data()[2] = harmmean;
  // (*top)[0]->mutable_cpu_data()[3] = precission;
  // (*top)[0]->mutable_cpu_data()[4] = f1_score; 
 
  int num = bottom[0]->num();

  for (int ind = 0; ind < num_labels; ++ind) {
      Dtype count_correct_cur = count_correct[ind] / num;
      (*top)[0]->mutable_cpu_data()[ind] = count_correct_cur;
  }

  // MultiLabelAccuracy should not be used as a loss function.
  return;
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
//  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
//      << "top_k must be less than or equal to the number of classes.";
  // Don't know how to deal with Reshape
  LOG(INFO)<<"MultiLabelAccuracyReshape Not Functioning";
//  CHECK_EQ(bottom[1]->channels(), 1);
//  CHECK_EQ(bottom[1]->height(), 1);
//  CHECK_EQ(bottom[1]->width(), 1);
//  (*top)[0]->Reshape(1, 1, 1, 1);
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);

}  // namespace caffe
