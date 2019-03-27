#ifndef CAFFE_UTIL_LIBDNNUTIL_H_
#define CAFFE_UTIL_LIBDNNUTIL_H_

#include "caffe/proto/caffe.pb.h"
#include <vector>
#include <iomanip>
#include <sstream>


namespace caffe{

class LibdnnInfo
{
public:
  LibdnnInfo(bool bias_term, int group, std::string name, std::string type,
  			 std::vector<int> in_shape, std::vector<int> out_shape,
  			 std::vector<int> kernel, std::vector<int> stride,
  			 std::vector<int> pad, std::vector<int> dilation) { 


  	name_ = name;

  	if(type == "Deconvolution") {
  		is_deconv = true;
  	}

	bias_term_ = bias_term;
	bias_multiplier_ = bias_term ? 1.0 : 0.0;
	int dims = in_shape.size();
	int spatial_dims = kernel.size();

	num_axes_ = spatial_dims;
	fmaps_in_ = in_shape[dims - spatial_dims - 1];
	fmaps_out_ = out_shape[dims - spatial_dims - 1];
	group_ = group;


	skip_range_check_ = true;

	for (int i = 0; i < spatial_dims; ++i) {
  	kernel_shape_.push_back(kernel[i]);
  	pad_.push_back(pad[i]);
  	if (pad_[i] > 0) {
  	  skip_range_check_ = false;
  	}
  	stride_.push_back(stride[i]);
  	dilation_.push_back(dilation[i]);
  	im_in_shape_.push_back(in_shape[dims - spatial_dims + i]);
  	im_out_shape_.push_back(out_shape[dims - spatial_dims + i]);
	}

  }
  ~LibdnnInfo() {}



  std::string generate_opencl_code() {
  	std::stringstream ss;

  	if(is_deconv) {
  		ss << deconv_fw_def() << std::endl;
  		ss << deconv_fw_kernel(name_ + "_forward") << std::endl;
  	} else {
  		ss << conv_fw_def() << std::endl;
  		ss << conv_fw_kernel(name_ + "_forward") << std::endl;
  	}

  	return ss.str();
  }

  


  std::vector<int> pad_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  std::vector<int> kernel_shape_;
  std::vector<int> im_in_shape_;
  std::vector<int> im_out_shape_;



  int num_axes_;
  int fmaps_in_;
  int fmaps_out_;
  int group_ = 1;
  bool bias_term_ = false;
  float bias_multiplier_;

  bool skip_range_check_;

  int vwm_ = 4;
  int vwn_ = 4;
  int tsk_unroll_ = 8;
  int wptm_ = 4;
  int wptn_ = 8;
  int rtsm_ = 4;
  int rtsn_ = 16;
  int tsk_ = 8;
  int tsm_ = wptm_ * rtsm_;
  int tsn_ = wptn_ * rtsn_;
  int lpta_ = (tsm_ * tsk_) / (rtsm_ * rtsn_);
  int lptb_ = (tsn_ * tsk_) / (rtsm_ * rtsn_);
  bool unroll_ = true;

  bool is_deconv = false;


  int M_FW_;
  int MG_FW_;
  int N_FW_;
  int K_FW_;
  int KG_FW_;

  std::string name_;


private:
  
  template<class T>
  void add_def(std::stringstream& ss,  // NOLINT
      const char* name, T value);

  template<class T>
  void add_def(std::stringstream& ss,  // NOLINT
      const std::string name, T value);

  std::string generate_gemm_core(bool dterm);
  std::string generate_accreg_init(bool dterm, bool load);


  std::string conv_fw_def();
  std::string conv_fw_kernel(std::string name);

  std::string deconv_fw_def();
  std::string deconv_fw_kernel(std::string name);
  
};

} //namespace caffe

#endif //CAFFE_UTIL_LIBDNNUTIL_H_