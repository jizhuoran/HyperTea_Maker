#ifndef HYPERTEA_UTIL_OPENCL_UTIL_H_
#define HYPERTEA_UTIL_OPENCL_UTIL_H_

#ifdef USE_OPENCL

#include <iostream>
#include <string.h>
#include <sstream>
#include <CL/cl.h>

#include "device_alternate.hpp"

namespace libdnn_tuner {


void cl_mem_destory(void* ptr);



class OpenCLHandler
{
public:

	~OpenCLHandler() {}
	

	static OpenCLHandler& Get();

	void DeviceQuery();
	void build_opencl_program(std::string kernel_code, cl_program &program);


  	std::string opencl_math_code(bool is_half);



  	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	  
	cl_context context;
	cl_command_queue commandQueue;

	cl_program tuning_program;


private:

	OpenCLHandler();
	OpenCLHandler(const OpenCLHandler&);
  	OpenCLHandler& operator=(const OpenCLHandler&);

};

}  // namespace libdnn_tuner

#endif //USE_OPENCL


#endif   // HYPERTEA_UTIL_OPENCL_UTIL_H_
