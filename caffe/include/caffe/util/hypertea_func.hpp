#ifndef CAFFE_UTIL_HYPERTEA_H_
#define CAFFE_UTIL_HYPERTEA_H_

#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <string.h>

#include "caffe/proto/caffe.pb.h"

#include "caffe/common.hpp"


namespace caffe {



inline std::string bool2string(bool b) { return b ? "true" : "false"; }


class FlyArray {

public:

  FlyArray(int length, const void* data_pointer, bool int_type = false);

  ~FlyArray() {}

  int length_;
  bool int_type_;
  void* data_pointer_;

  std::string dtype_;
  int dtype_size_;



};

class FlyDefs
{
public:

	FlyDefs(std::string op_type, std::string op_name,
			std::string cpu_signature,
			std::string gpu_signature) : 
	op_type_(op_type), op_name_(op_name),
	cpu_signature_(cpu_signature),
	gpu_signature_(gpu_signature) { }
	~FlyDefs() { }
	

	std::string op_type_, op_name_;
	std::string cpu_signature_, gpu_signature_;


};




class hypertea_func
{
public:

	static hypertea_func& Get();

	~hypertea_func() {}


	std::vector<std::string> input_names;
	std::vector<std::string> output_names;


	void append_op_defs(std::string type, std::string name, std::string cpu_signature, std::string gpu_signature);
	void append_op_params(std::vector<std::pair<std::string, FlyArray>> parameter);
	void append_op_inoutputs(std::vector<std::pair<std::string, FlyArray>> inoutputs);
	void append_op_inference(std::string inference_signature);

	std::string hypertea_cpu(std::string net_name);
	std::string hypertea_gpu(std::string net_name);

	std::stringstream hypertea_libdnn;

	void set2half() {dtype_ = "half"; dtype_size_ = sizeof(half);}
	void set2float() {dtype_ = "float"; dtype_size_ = sizeof(float);}

	std::string dtype_;
	int dtype_size_;

private:


	std::string inference_signature();

	std::string gpu_param_defs();
	std::string cpu_param_defs();

	std::string gpu_param_free();
	void save_conv_file(std::string net_name);

	std::string copy_data_cpu(bool from_user);
	std::string copy_data_gpu(bool from_user);

	std::string cpu_free_inoutputs();
	std::string gpu_free_inoutputs();

	std::string cpu_inoutputs_defs();
	std::string gpu_inoutputs_defs();

	std::string cpu_op_defs();
	std::string gpu_op_defs();


	std::string cpu_op_runs();
	std::string gpu_op_runs();

	std::string copy_weight_cpu(std::string net_name);
	std::string copy_weight_gpu(std::string net_name);

	// std::stringstream op_run;
	std::vector<std::string> op_run;

	std::vector<FlyDefs> hypertea_op_defs;
	std::map<std::string, FlyArray> hypertea_inoutputs;
	std::vector<std::pair<std::string, FlyArray>> hypertea_params;


	hypertea_func() : dtype_("float"), dtype_size_(sizeof(float)) {}
  	hypertea_func(const hypertea_func&);
    hypertea_func& operator=(const hypertea_func&);
	
};






}


#endif