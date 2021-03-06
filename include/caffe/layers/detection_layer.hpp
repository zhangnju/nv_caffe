#ifndef CAFFE_DETECTION_LAYER_HPP_
#define CAFFE_DETECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {

template <typename Ftype, typename Btype>
class DetectionLayer : public Layer<Ftype, Btype> {
 public:
  explicit DetectionLayer(const LayerParameter& param)
	  : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Detection"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual void Backward_cpu(const vector<Blob*>& top,
	  const vector<bool>& propagate_down, const vector<Blob*>& bottom){
	  for (int i = 0; i < propagate_down.size(); ++i) {
		  if (propagate_down[i]) { NOT_IMPLEMENTED; }
	  }
  }

  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual void Backward_gpu(const vector<Blob*>& top,
	  const vector<bool>& propagate_down, const vector<Blob*>& bottom){
	  for (int i = 0; i < propagate_down.size(); ++i) {
		  if (propagate_down[i]) { NOT_IMPLEMENTED; }
	  }
  }

  int width_;
  int height_;
  int coords_;
  int softmax_;
  int batch_;
  int num_class_;
  int num_object_;
  float threshold_;
  bool sqrt_;
  bool constriant_;
  int score_type_;
  float thresh_;
};

}  // namespace caffe

#endif  // CAFFE_DETECTION_LAYER_HPP_
