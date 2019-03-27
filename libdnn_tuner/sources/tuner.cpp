#include <numeric>
#include <sstream>
#include <map>
#include <cmath>

#include "opencl_util.hpp"
#include "libdnn_util.hpp"
#include "benchmark.hpp"
#include "glog_wrapper.hpp"
#include "tuner.hpp"

const int MAX_ALUS = 64;
const int MIN_ALUS = 16;
const bool verbose = false;
const int TEST_TIME = 3;

std::string var2string(const std::vector<int> v) {

  std::stringstream ss;
  
  ss << v[0];
  for (int i = 1; i < v.size(); ++i) {
    ss << ", " << v[i];
  }

  return ss.str();
}



std::vector<std::vector<int> > generate_parameter_list(int M, int N, int K) {

	std::map<std::string, int> name2index = {
		{"VWM", 0},
		{"VMN", 1},
		{"TSK_UNROLL", 2},
		{"WPTM", 3},
		{"WPTN", 4},
		{"RTSM", 5},
		{"RTSN", 6},
		{"TSK", 7}
	};

	std::vector<std::vector<int> > tune_parameters = {
		std::vector<int> {2, 4}, //vwm
		std::vector<int> {2, 4, 8}, //vwn
		std::vector<int> {1}, //tsk_unroll
		std::vector<int> {1, 2, 4, 8, 16}, //wptm
		std::vector<int> {4, 8, 16, 32}, //wptn
		std::vector<int> {1, 2, 4, 8, 16}, //rtsm
		std::vector<int> {4, 8, 16, 32}, //rtsn
		std::vector<int> {8, 16, 24, 32} //tsk
	};

	std::vector<std::vector<int> > parameters_list;

	for(auto const & v0:tune_parameters[0])
	for(auto const & v1:tune_parameters[1])
	for(auto const & v2:tune_parameters[2])
	for(auto const & v3:tune_parameters[3])
	for(auto const & v4:tune_parameters[4])
	for(auto const & v5:tune_parameters[5])
	for(auto const & v6:tune_parameters[6])
	for(auto const & v7:tune_parameters[7]) {

		std::vector<int> parameters{v0, v1, v2, v3, v4, v5, v6, v7};

		bool valid_combination = (
			(parameters[name2index["TSK"]] * parameters[name2index["WPTM"]]) % parameters[name2index["RTSM"]] == 0 &&
			(parameters[name2index["TSK"]] * parameters[name2index["WPTN"]]) % parameters[name2index["RTSN"]] == 0 &&
			parameters[name2index["TSK"]] % parameters[name2index["TSK_UNROLL"]] == 0 &&
			parameters[name2index["VWM"]] % parameters[name2index["TSK_UNROLL"]] == 0 &&
			parameters[name2index["VWN"]] % parameters[name2index["TSK_UNROLL"]] == 0 &&
			parameters[name2index["WPTM"]] % parameters[name2index["VWM"]] == 0 &&
			parameters[name2index["WPTN"]] % parameters[name2index["VWN"]] == 0
		);


		valid_combination &= (parameters[name2index["RTSM"]] * parameters[name2index["RTSN"]] <= MAX_ALUS);
		valid_combination &= (parameters[name2index["RTSM"]] * parameters[name2index["RTSN"]] >= MIN_ALUS);
		

		valid_combination &= (parameters[name2index["RTSM"]] * parameters[name2index["WPTM"]] <= 2*pow(2, ceil(log(M)/log(2))));
		valid_combination &= (parameters[name2index["RTSN"]] * parameters[name2index["WPTN"]] <= 2*pow(2, ceil(log(N)/log(2))));


		valid_combination &= (parameters[name2index["RTSM"]] * parameters[name2index["WPTM"]] >= ((M > 8)? 8:M));
		valid_combination &= (parameters[name2index["RTSN"]] * parameters[name2index["WPTN"]] >= ((N > 8)? 8:N));


		valid_combination &= (parameters[name2index["TSK"]] * parameters[name2index["TSM"]] >= parameters[name2index["RTSM"]] * parameters[name2index["RTSN"]]);
		valid_combination &= (parameters[name2index["TSK"]] * parameters[name2index["TSN"]] >= parameters[name2index["RTSM"]] * parameters[name2index["RTSN"]]);




		int memeory_usage =  parameters[name2index["TSK"]] * parameters[name2index["WPTN"]] * parameters[name2index["RTSN"]]
					     	+ parameters[name2index["TSK"]] * parameters[name2index["WPTM"]] * parameters[name2index["RTSM"]];


		valid_combination &= (memeory_usage <= 1024);

		if(valid_combination) {
			parameters_list.push_back(parameters);
		}
	}

	return parameters_list;
}


std::string profile_this_config(
	libdnn_tuner::libdnn_setting setting,
	cl_mem input_data,
	cl_mem weight,
	cl_mem bias,
	cl_mem output_data,
	std::vector<int> bottom_shape,
	std::vector<int> top_shape,
	std::vector<int> kernel_vec,
	std::vector<int> stride_vec,
	std::vector<int> pad_vec,
	std::vector<int> dilation_vec) {


	libdnn_tuner::GPUTimer timer;

	libdnn_tuner::LibdnnInfo info("this_bias", 1, "conv1", "Deconvolution",
              bottom_shape, top_shape,
              kernel_vec, stride_vec, 
              pad_vec, dilation_vec,
              setting);


	cl_program tuning_program;


	libdnn_tuner::OpenCLHandler::Get().build_opencl_program(info.generate_opencl_code(), tuning_program);

  	cl_int ret = -1;
  	cl_kernel kernel = clCreateKernel(tuning_program, "conv1_forward", &ret);
  	OPENCL_CHECK(ret);


		
	OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_data));  
	OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&weight));  
	OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_data));

	if (bias) {
   		OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bias));
	}

	int global_size_x = ((((top_shape[2] * top_shape[3]) - 1) / info.tsn_ + 1)*info.rtsn_);
	int global_size_y = ((((top_shape[1] / info.group_) - 1) / info.tsm_ + 1)*info.rtsm_);
	int global_size_z = (bottom_shape[0] * info.group_);


	size_t* local_size = new size_t[3];
	local_size[0] = info.rtsn_, info.rtsm_, 1;
	local_size[1] = info.rtsm_;
	local_size[2] = 1;

	size_t* global_size = new size_t[3];
	global_size[0] = global_size_x;
	global_size[1] = global_size_y;
	global_size[2] = global_size_z;

	float elpased_time = 0;


	for (int i = 0; i < TEST_TIME; ++i) {
		timer.Start();
			OPENCL_CHECK(clEnqueueNDRangeKernel(libdnn_tuner::OpenCLHandler::Get().commandQueue, kernel, 3, NULL, global_size, local_size, 0, NULL, NULL));  
		timer.Stop();

		elpased_time += timer.MilliSeconds();
	}
	
	elpased_time /= TEST_TIME;
	
	if (verbose) {
		LOG(INFO) << std::endl
			  << "------------------------------------" << std::endl 
	          << "SIGNATURE: " << std::endl
			  << "	bottom_shape: " << var2string(bottom_shape) << std::endl
	          << "	top_shape: " << var2string(top_shape) << std::endl
    		  << "	kernel_vec: "  << var2string(kernel_vec) << std::endl
    		  << "	stride_vec: "  << var2string(stride_vec) << std::endl
    		  << "	pad_vec: "  << var2string(pad_vec) << std::endl
    		  << "	dilation_vec: "  << var2string(dilation_vec) << std::endl
    		  << std::endl
    		  << "Config:" << std::endl
			  << "	vwm: " << setting.vwm_ << std::endl
			  << "	vwn: " << setting.vwn_ << std::endl
			  << "	tsk_unroll: " << setting.tsk_unroll_ << std::endl
			  << "	wptm: " << setting.wptm_ << std::endl
			  << "	wptn: " << setting.wptn_ << std::endl
			  << "	rtsm: " << setting.rtsm_ << std::endl
			  << "	rtsn: " << setting.rtsn_ << std::endl
			  << "	tsk: " << setting.tsk_ << std::endl
			  << "	tsm: " << setting.tsm_ << std::endl
			  << "	tsn: " << setting.tsn_ << std::endl
			  << "	lpta: " << setting.lpta_ << std::endl
			  << "	lptb: " << setting.lptb_ << std::endl
			  << std::endl
			  << "The time usage is: " << elpased_time << "ms." << std::endl
			  << std::endl
			  << "Global Size " << var2string(std::vector<int>{global_size_x, global_size_y, global_size_z}) << std::endl
			  << "Local Size " << var2string(std::vector<int>{info.rtsn_, info.rtsm_, 1}) << std::endl
			  << "------------------------------------" << std::endl 
			  << std::endl << std::endl;
	}


  	std::stringstream ss;

  	ss << setting.vwm_ << ", "
	   << setting.vwn_ << ", "
	   << setting.tsk_unroll_ << ", "
	   << setting.wptm_ << ", "
	   << setting.wptn_ << ", "
	   << setting.rtsm_ << ", "
	   << setting.rtsn_ << ", "
	   << setting.tsk_ << ", "
	   << setting.tsm_ << ", "
	   << setting.tsn_ << ", "
	   << setting.lpta_ << ", "
	   << setting.lptb_ << ", "
	   << elpased_time;

  	return ss.str();

}




void profile() {


	std::vector<int> bottom_shape{1, 32, 512, 512};
	std::vector<int> top_shape{1, 3, 512, 512};
	std::vector<int> kernel_vec{9, 9};
	std::vector<int> stride_vec{1, 1};
	std::vector<int> pad_vec{4, 4};
	std::vector<int> dilation_vec{1, 1};


	int M = top_shape[1];
	int K = kernel_vec[0] * kernel_vec[1] * bottom_shape[1];
	int N = top_shape[2] * top_shape[3];



	size_t input_count = std::accumulate(bottom_shape.begin(), bottom_shape.end(), 1, std::multiplies<int>());
  	size_t output_count = std::accumulate(top_shape.begin(), top_shape.end(), 1, std::multiplies<int>());
  	size_t weight_count = bottom_shape[1] * top_shape[1] * std::accumulate(kernel_vec.begin(), kernel_vec.end(), 1, std::multiplies<int>());
  	size_t bias_count = top_shape[1];
  	cl_mem input_data = clCreateBuffer(libdnn_tuner::OpenCLHandler::Get().context, CL_MEM_READ_WRITE, input_count * sizeof(float), NULL, NULL);
	cl_mem weight = clCreateBuffer(libdnn_tuner::OpenCLHandler::Get().context, CL_MEM_READ_WRITE, weight_count * sizeof(float), NULL, NULL);
	cl_mem bias = clCreateBuffer(libdnn_tuner::OpenCLHandler::Get().context, CL_MEM_READ_WRITE, bias_count * sizeof(float), NULL, NULL);
	cl_mem output_data = clCreateBuffer(libdnn_tuner::OpenCLHandler::Get().context, CL_MEM_READ_WRITE, output_count * sizeof(float), NULL, NULL);


	LOG(INFO) << "vwm, vwn, tsk_unroll, wptm, wptn, rtsm, rtsn, tsk, tsm, tsn, lpta, lptb, time";

	auto parameters_list = generate_parameter_list(M, N, K);

	for(auto const & v: parameters_list) {
		libdnn_tuner::libdnn_setting setting(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], true); //vwm, vwn, tsk_unroll, wptm, wptn, rtsm, rtsn, tsk, unroll
		
		auto profile_result = profile_this_config(
			setting, 
			input_data, 
			weight, 
			bias, 
			output_data,
			bottom_shape,
			top_shape,
			kernel_vec,
			stride_vec,
			pad_vec,
			dilation_vec
		);
		
		std::cout << profile_result << std::endl;
	}

  	clReleaseMemObject(input_data);
  	clReleaseMemObject(output_data);
  	clReleaseMemObject(weight);
  	clReleaseMemObject(bias);

}



