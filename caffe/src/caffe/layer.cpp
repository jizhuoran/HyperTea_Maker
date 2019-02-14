#include "caffe/layer.hpp"

namespace caffe {
	
	
	template <>
	std::vector<std::pair<std::string, FlyArray> > Layer<float>::get_inoutputs(
														const vector<Blob<float>*>& bottom,
    													const vector<Blob<float>*>& top) {

		std::vector<std::pair<std::string, FlyArray> > hypertea_inoutputs;

		for (int i = 0; i < this->layer_param_.top_size(); ++i) {
			hypertea_inoutputs.push_back(
  			std::pair<std::string, FlyArray>(this->layer_param_.top(i), 
  				FlyArray(top[i]->count(), NULL)));
		}

		for (int i = 0; i < this->layer_param_.bottom_size(); ++i) {
			hypertea_inoutputs.push_back(
  			std::pair<std::string, FlyArray>(this->layer_param_.bottom(i), 
  				FlyArray(bottom[i]->count(), NULL)));
		}

		return hypertea_inoutputs;

	}

	template <>
	std::string Layer<float>::get_forward_signature() {

		std::stringstream ss;
  
		ss << this->layer_param_.name() << ".Forward("; 

		ss << "{" << this->layer_param_.bottom(0) << "_data";
		for (int i = 1; i < this->layer_param_.bottom_size(); ++i) {
		  ss << " , " << this->layer_param_.bottom(i) << "_data";
		}
		ss << "}";

		ss << ", ";

		ss << "{" << this->layer_param_.top(0) << "_data";
		for (int i = 1; i < this->layer_param_.top_size(); ++i) {
		  ss << " , " << this->layer_param_.top(i) << "_data";
		}
		ss << "});" << std::endl;

		return ss.str();
	}

	template <>
	void Layer<float>::produce_hypertea(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {

		if (!this->layer_param_.type().compare("Input")) {
    		return;
  		}

  		this->hypertea_hw();

  		hypertea_func::Get().append_op_defs(this->type(), this->layer_param_.name(),
  										get_cpu_signature(bottom, top), get_gpu_signature(bottom, top));
		hypertea_func::Get().append_op_params(this->get_params(bottom, top));
		hypertea_func::Get().append_op_inoutputs(this->get_inoutputs(bottom, top));
		hypertea_func::Get().append_op_inference(this->get_forward_signature());
	

	}



	template <>
	void Layer<double>::produce_hypertea(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {

		LOG(FATAL) << "We do not support double now!!!" << std::endl;

	}








INSTANTIATE_CLASS(Layer);


}  // namespace caffe
