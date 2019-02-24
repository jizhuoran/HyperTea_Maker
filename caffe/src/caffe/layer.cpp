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
  
		// ss << this->layer_param_.name() << ".Forward("; 

		// ss << "{" << this->layer_param_.bottom(0) << "_data.data()";
		// for (int i = 1; i < this->layer_param_.bottom_size(); ++i) {
		//   ss << " , " << this->layer_param_.bottom(i) << "_data.data()";
		// }
		// ss << "}";

		// ss << ", ";

		// ss << "{" << this->layer_param_.top(0) << "_data.data()";
		// for (int i = 1; i < this->layer_param_.top_size(); ++i) {
		//   ss << " , " << this->layer_param_.top(i) << "_data.data()";
		// }
		// ss << "});" << std::endl;


		ss << this->layer_param_.top(0) << "_data = "
		   << this->layer_param_.name() << "("
		   << this->layer_param_.bottom(0) << "_data);" 
		   << std::endl;

		return ss.str();
	}

	template <>
	FlySignature Layer<float>::get_forward_signature_general() {

		std::string layer_name = this->layer_param_.name(); 

		std::vector<std::string> inputs_name;
		for (int i = 0; i < this->layer_param_.bottom_size(); ++i) {
		  inputs_name.push_back(this->layer_param_.bottom(i));
		}

		std::vector<std::string> outputs_name;
		for (int i = 0; i < this->layer_param_.top_size(); ++i) {
		  outputs_name.push_back(this->layer_param_.top(i));
		}

		return FlySignature(this->type(), layer_name, inputs_name, outputs_name);
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
		hypertea_func::Get().append_op_inference(this->get_forward_signature_general());
	

	}



	template <>
	void Layer<double>::produce_hypertea(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {

		LOG(FATAL) << "We do not support double now!!!" << std::endl;

	}








INSTANTIATE_CLASS(Layer);


}  // namespace caffe
