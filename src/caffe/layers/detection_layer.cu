#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/layer.hpp"
#include "caffe/layers/detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ 
void detection_kernel(const int width_, const int height_, const int num_object_,
    const int num_class_, const int coords_,const int sqrt_,const float thresh_,Dtype* input_data, Dtype* box_data,Dtype* prob_data) 
    {
    for (int i = 0; i < width_*height_; ++i){
	  int row = i / width_;
	  int col = i % width_;
	  for (int n = 0; n < num_object_; ++n){
		  int index = i*num_object_ + n;
		  int p_index = width_*height_*num_class_ + i*num_object_ + n;
		  Dtype scale = input_data[p_index];
		  int box_index = width_*height_*(num_class_ + num_object_) + (i*num_object_ + n) * 4;
		  box_data[index*coords_] = (input_data[box_index + 0] + col) / width_; //check me ,here need to multiplied by image width
		  box_data[index*coords_ + 1] = (input_data[box_index + 1] + row) / width_;//check me ,here need to multiplied by image height
		  box_data[index*coords_ + 2] = pow(input_data[box_index + 2], Dtype((sqrt_ ? 2 : 1)));
		  box_data[index*coords_ + 3] = pow(input_data[box_index + 3], Dtype((sqrt_ ? 2 : 1)));
		  Dtype max_prob = 0;
		  for (int j = 0; j < num_class_; ++j){
			  int class_index = i*num_class_;
			  Dtype prob = scale*input_data[class_index + j];
			  prob_data[index*(num_class_+1) + j] = (prob > Dtype(thresh_)) ? prob : Ftype(0);
			  if (prob > max_prob) max_prob = prob;
		  }
		  prob_data[index*(num_class_ + 1) + num_class_] = max_prob>Dtype(thresh_) ? max_prob : Ftype(0);
	  }
  }
}

template <typename Ftype, typename Btype>
void DetectionLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  Ftype* input_data = bottom[0]->mutable_gpu_data<Ftype>();
  Ftype* box_data = top[0]->mutable_gpu_data<Ftype>();//check the size is right
  Ftype* prob_data = top[1]->mutable_gpu_data<Ftype>();
  /*
  if (softmax_){
	  for (int b = 0; b < batch_; ++b){
		  int index = b*width_*height_*((1 + coords_)*num_object_ + num_class_);
		  for (int i = 0; i < width_*height_; ++i) {
			  int offset = i*num_class_;
			  softmax_op(input_data + index + offset, num_class_,1);
		  }
	  }
  }
  */
  dim3 dimBlock( 1, 1 );
  dim3 dimGrid( 1, 1 );
  detection_kernel<<<dimGrid, dimBlock>>>(width_,height_,num_object_,num_class_,coords_,sqrt_,thresh_,input_data,box_data,prob_data);
  
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(DetectionLayer);
}  // namespace caffe
