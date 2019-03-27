#ifndef CAFFE_UTIL_LIBDNNUTIL_H_
#define CAFFE_UTIL_LIBDNNUTIL_H_

#include <vector>
#include <iomanip>
#include <sstream>


namespace libdnn_tuner{

class libdnn_setting
{

public:
  libdnn_setting(
    int vwm, int vwn, 
    int tsk_unroll, 
    int wptm, int wptn, 
    int rtsm, int rtsn, 
    int tsk, bool unroll) 

  : vwm_(vwm), vwn_(vwn), 
  tsk_unroll_(tsk_unroll), 
  wptm_(wptm), wptn_(wptn), 
  rtsm_(rtsm), rtsn_(rtsn), 
  tsk_(tsk), unroll_(unroll) {

    tsm_ = wptm_ * rtsm_;
    tsn_ = wptn_ * rtsn_;
    lpta_ = (tsm_ * tsk_) / (rtsm_ * rtsn_);
    lptb_ = (tsn_ * tsk_) / (rtsm_ * rtsn_);


  }


  


  ~libdnn_setting() = default;

  bool unroll_;

  int vwm_;
  int vwn_;
  int tsk_unroll_;
  int wptm_;
  int wptn_;
  int rtsm_;
  int rtsn_;
  int tsk_;
  int tsm_;
  int tsn_;
  int lpta_;
  int lptb_;
  
};


class LibdnnInfo
{
public:
  LibdnnInfo(bool bias_term, int group, std::string name, std::string type,
  			 std::vector<int> in_shape, std::vector<int> out_shape,
  			 std::vector<int> kernel, std::vector<int> stride,
  			 std::vector<int> pad, std::vector<int> dilation,
         libdnn_setting setting) :

  name_(name),
  bias_term_(bias_term),


  unroll_(setting.unroll_),
  vwm_(setting.vwm_),
  vwn_(setting.vwn_),
  tsk_unroll_(setting.tsk_unroll_),
  wptm_(setting.wptm_),
  wptn_(setting.wptn_),
  rtsm_(setting.rtsm_),
  rtsn_(setting.rtsn_),
  tsk_(setting.tsk_),
  tsm_(setting.tsm_),
  tsn_(setting.tsn_),
  lpta_(setting.lpta_),
  lptb_(setting.lptb_) { 



	if(type == "Deconvolution") {
		is_deconv = true;
	}

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

    ss << "#define Dtype float" << std::endl;
    ss << "#define Dtype1 float" << std::endl;
    ss << "#define Dtype2 float2" << std::endl;
    ss << "#define Dtype4 float4" << std::endl;
    ss << "#define Dtype8 float8" << std::endl;
    ss << "#define Dtype16 float16" << std::endl;
    ss << "#define VEC_1_0(X) X" << std::endl;
    ss << "#define VEC_2_0(X) X.x" << std::endl;
    ss << "#define VEC_2_1(X) X.y" << std::endl;
    ss << "#define VEC_4_0(X) X.x" << std::endl;
    ss << "#define VEC_4_1(X) X.y" << std::endl;
    ss << "#define VEC_4_2(X) X.z" << std::endl;
    ss << "#define VEC_4_3(X) X.w" << std::endl;
    ss << "#define VEC_8_0(X) X.s0" << std::endl;
    ss << "#define VEC_8_1(X) X.s1" << std::endl;
    ss << "#define VEC_8_2(X) X.s2" << std::endl;
    ss << "#define VEC_8_3(X) X.s3" << std::endl;
    ss << "#define VEC_8_4(X) X.s4" << std::endl;
    ss << "#define VEC_8_5(X) X.s5" << std::endl;
    ss << "#define VEC_8_6(X) X.s6" << std::endl;
    ss << "#define VEC_8_7(X) X.s7" << std::endl;
    ss << "#define VEC_16_0(X) X.s0" << std::endl;
    ss << "#define VEC_16_1(X) X.s1" << std::endl;
    ss << "#define VEC_16_2(X) X.s2" << std::endl;
    ss << "#define VEC_16_3(X) X.s3" << std::endl;
    ss << "#define VEC_16_4(X) X.s4" << std::endl;
    ss << "#define VEC_16_5(X) X.s5" << std::endl;
    ss << "#define VEC_16_6(X) X.s6" << std::endl;
    ss << "#define VEC_16_7(X) X.s7" << std::endl;
    ss << "#define VEC_16_8(X) X.s8" << std::endl;
    ss << "#define VEC_16_9(X) X.s9" << std::endl;
    ss << "#define VEC_16_10(X) X.sA" << std::endl;
    ss << "#define VEC_16_11(X) X.sB" << std::endl;
    ss << "#define VEC_16_12(X) X.sC" << std::endl;
    ss << "#define VEC_16_13(X) X.sD" << std::endl;
    ss << "#define VEC_16_14(X) X.sE" << std::endl;
    ss << "#define VEC_16_15(X) X.sF" << std::endl;



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

  

  int vwm_;
  int vwn_;
  int tsk_unroll_;
  int wptm_;
  int wptn_;
  int rtsm_;
  int rtsn_;
  int tsk_;
  int tsm_;
  int tsn_;
  int lpta_;
  int lptb_;



  bool unroll_;

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

} //namespace libdnn_tuner

#endif //CAFFE_UTIL_LIBDNNUTIL_H_