#include "caffe/util/hypertea_func.hpp"
#include <boost/thread.hpp>

#include <sstream>

namespace caffe {



FlyArray::FlyArray(int length, const void* data_pointer, bool int_type) {

  length_ = length;

  if (data_pointer != NULL) {
    data_pointer_ = malloc(sizeof(float) * length);
    memcpy(data_pointer_, data_pointer, sizeof(float) * length);
  }

  dtype_ = int_type? "int" : hypertea_func::Get().dtype_;
  dtype_size_ = int_type? sizeof(int) : hypertea_func::Get().dtype_size_;
}




static boost::thread_specific_ptr<hypertea_func> hypertea_thread_instance_;

hypertea_func& hypertea_func::Get() {
  if (!hypertea_thread_instance_.get()) {
    hypertea_thread_instance_.reset(new hypertea_func());
  }
  return *(hypertea_thread_instance_.get());
}

void hypertea_func::append_op_defs(std::string type, std::string name, std::string cpu_signature, std::string gpu_signature) {

  hypertea_op_defs.push_back(FlyDefs(type, name, cpu_signature, gpu_signature));

}

void hypertea_func::append_op_params(std::vector<std::pair<std::string, FlyArray>> parameter) {

    for(auto const& param: parameter) {
      hypertea_params.push_back(param);
    }
}


void hypertea_func::append_op_inoutputs(std::vector<std::pair<std::string, FlyArray>> inoutputs) {

    for(auto const& inoutput: inoutputs) {
      hypertea_inoutputs.insert(inoutput);
    }
}


void hypertea_func::append_op_inference(std::string inference_signature) {

  op_run.push_back(inference_signature);

}






std::string hypertea_func::cpu_op_defs() { 
  
  std::stringstream ss;
  
  for(auto const& def:hypertea_op_defs) {
    ss << def.op_type_ << "Op_CPU<" << this->dtype_ << "> " << def.op_name_ << " = "
       << def.op_type_ << "Op_CPU<" << this->dtype_ << ">( "
       << def.cpu_signature_ << ");" << std::endl;
  }
  
  return ss.str();

}

std::string hypertea_func::gpu_op_defs() { 
  
  std::stringstream ss;
  
  for(auto const& def:hypertea_op_defs) {
    ss << def.op_type_ << "Op_GPU<" << this->dtype_ << "> " << def.op_name_ << " = "
       << def.op_type_ << "Op_GPU<" << this->dtype_ << ">( "
       << def.gpu_signature_ << ");" << std::endl;
  }
  
  return ss.str();

}


std::string hypertea_func::cpu_op_runs() { 
  
  std::stringstream ss;
  
  for(auto const& op_para:op_run) {
    ss << op_para << std::endl;

  }
  
  return ss.str();

}

std::string hypertea_func::gpu_op_runs() { 
  
  return cpu_op_runs();

}



std::string hypertea_func::copy_weight_cpu(std::string net_name) {

  std::stringstream ss_head;
  std::stringstream ss_body;

  FILE *f = fopen((net_name + ".weight").c_str(), "wb");

  long weight_begin_pos = 0;

  for (auto const & weight_item : hypertea_params) {

    auto fly_weight = weight_item.second;

    fwrite(fly_weight.data_pointer_, fly_weight.dtype_size_, fly_weight.length_, f);

    weight_begin_pos = ftell(f);

  }

  fclose(f);

  ss_head << "FILE *f = fopen(" << "\"" << (net_name + ".weight").c_str() << "\"" << ", \"rb\");" << std::endl
          << "size_t read_size = fread(all_weights, 1, weight_size, f);" << std::endl
          << "if (read_size != weight_size) {"
          << "  LOG(ERROR) << \"Weight File Size Mismatch\" << read_size << \" and \" << weight_size << std::endl;" << std::endl
          << "}" << std::endl
          << "fclose(f);" << std::endl << std::endl;


  return ss_head.str() + ss_body.str();
}


std::string hypertea_func::copy_weight_gpu(std::string net_name) {


  std::stringstream ss_body;
  std::stringstream ss_head;

  FILE *f = fopen((net_name + ".weight").c_str(), "wb");
  long weight_begin_pos = 0;

  for (auto const & weight_item : hypertea_params) {

    auto weight_name = weight_item.first;
    auto fly_weight = weight_item.second;

    fwrite(fly_weight.data_pointer_, fly_weight.dtype_size_, fly_weight.length_, f);

    ss_body << "OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, "
            << weight_item.first  << ", "
            << "CL_TRUE, 0, "
            << fly_weight.length_ * fly_weight.dtype_size_ << ", "
            << "all_weights + " << weight_begin_pos << ", "
            << "0, NULL, NULL));" << std::endl;


    weight_begin_pos = ftell(f);
  }
  
  fclose(f);


  ss_head << "int weight_size = " << weight_begin_pos << ";" << std::endl
          << "unsigned char* all_weights = (unsigned char*) malloc(weight_size);" << std::endl << std::endl
          << "FILE *f = fopen(" << "\"" << (net_name + ".weight").c_str() << "\"" << ", \"rb\");" << std::endl
          << "size_t read_size = fread(all_weights, 1, weight_size, f);" << std::endl
          << "if (read_size != weight_size) {"
          << "  LOG(ERROR) << \"Weight File Size Mismatch\" << read_size << \" and \" << weight_size << std::endl;" << std::endl
          << "}" << std::endl
          << "fclose(f);" << std::endl << std::endl;

  return ss_head.str() + ss_body.str() + "free(all_weights);\n\n";

}


std::string hypertea_func::gpu_param_defs() {

  std::stringstream ss;

  for (auto const & weight_item : hypertea_params) {

      auto fly_weight = weight_item.second;

      ss << "cl_mem " << weight_item.first << " = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, "
        << fly_weight.length_ * fly_weight.dtype_size_ << ", "
        << "NULL" << ", "
        << "NULL" << ");" << std::endl;

    }

  return ss.str();
}



std::string hypertea_func::cpu_param_defs() {

  std::stringstream ss_head;
  std::stringstream ss_body;


  long weight_size = 0;

  for (auto const & weight_item : hypertea_params) {

    auto fly_weight = weight_item.second;

    ss_body << fly_weight.dtype_ << "* " 
            << weight_item.first << " = "
            << "reinterpret_cast<" << fly_weight.dtype_ << "*>"
            << "(all_weights + " << weight_size << ");" << std::endl;

    weight_size += fly_weight.dtype_size_ * fly_weight.length_;

  }


  ss_head << "int weight_size = " << weight_size << ";" << std::endl
          << "unsigned char* all_weights = (unsigned char*) malloc(weight_size);" << std::endl << std::endl;

  return ss_head.str() + ss_body.str();

}




std::string hypertea_func::gpu_param_free() {

  std::stringstream ss;

  for (auto const & weight_item : hypertea_params) {

      ss << "OPENCL_CHECK(clReleaseMemObject(" << weight_item.first << "));" << std::endl;

    }

  return ss.str();
}


void hypertea_func::save_conv_file(std::string net_name) {

  std::stringstream header_ss;
  if (dtype_ == "float") {
    header_ss << "#define Dtype float" << std::endl;
    header_ss << "#define Dtype1 float" << std::endl;
    // float2, float4, float8, float16
    for (int i = 2; i <= 16; i *= 2) {
      header_ss << "#define Dtype" << i << " float" << i << std::endl;
    }
  } else if (dtype_ == "half"){

    header_ss << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << std::endl;
    header_ss << "#define Dtype half" << std::endl;
    header_ss << "#define Dtype1 half" << std::endl;
    // half2, half4, half8, half16
    for (int i = 2; i <= 16; i *= 2) {
      header_ss << "#define Dtype" << i << " half" << i << std::endl;
    }
  }

  std::vector<std::string> elems4({
      "x", "y", "z", "w" });
  std::vector<std::string> elems16({
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
      "s8", "s9", "sA", "sB", "sC", "sD", "sE", "sF" });

  for (int i = 1; i <= 16; i *= 2) {
    for (int j = 0; j < i; ++j) {
      if (i == 1) {
        header_ss << "#define VEC_" << i << "_" << j << "(X)" << " X" << std::endl;
      } else if (i < 8) {
        header_ss << "#define VEC_" << i << "_" << j << "(X)" << " X." << elems4[j]
           << std::endl;
      } else {
        header_ss << "#define VEC_" << i << "_" << j << "(X)" << " X." << elems16[j]
           << std::endl;
      }
    }
  }


  std::ofstream conv_cl_file (net_name + "_conv_opencl.hpp");
  if (conv_cl_file.is_open()) {
    conv_cl_file << "std::string conv_opencl_funcs = R\"(";
    conv_cl_file << header_ss.str() << std::endl;
    conv_cl_file << hypertea_libdnn.str();
    conv_cl_file << ")\";" << std::endl;
  }

}






std::string hypertea_func::cpu_inoutputs_defs() {

    std::stringstream ss;

    for (auto const & inoutput : hypertea_inoutputs) {
      
      std::string name = inoutput.first;
      auto info = inoutput.second;
      
      ss << this->dtype_ << "* " << name << "_data = ("
         << this->dtype_ << " *)malloc(" << this->dtype_size_ * info.length_ << ");" << std::endl;

    }

    return ss.str();
}


std::string hypertea_func::gpu_inoutputs_defs() {

    std::stringstream ss;

    for (auto const & inoutput : hypertea_inoutputs) {
      
      std::string name = inoutput.first;
      auto info = inoutput.second;
      
      ss << "cl_mem " << name << "_data = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, "
              << this->dtype_size_ * info.length_ << ", "
              << "NULL" << ", "
              << "NULL" << ");" << std::endl;
    }

    return ss.str();
}


std::string hypertea_func::cpu_free_inoutputs() {


    std::stringstream ss;

    for (auto const & inoutput : hypertea_inoutputs) {
      
      std::string name = inoutput.first;
      
      ss << "free(" << name + "_data" << ");" << std::endl;

    }

    return ss.str();
}

std::string hypertea_func::gpu_free_inoutputs() {

    std::stringstream ss;

    for (auto const & inoutput : hypertea_inoutputs) {
      
      std::string name = inoutput.first; 
      ss << "OPENCL_CHECK(clReleaseMemObject(" << name + "_data" << "));" << std::endl;

    }

    return ss.str();
}




std::string hypertea_func::inference_signature() {

  std::stringstream ss;
  
  for (auto const& input_name : input_names) {
    ss << "std::vector<" << this->dtype_ << "> &" << input_name << "_from_user" << ", ";
  }

  for (auto const& output_name : output_names) {
    ss << "std::vector<" << this->dtype_ << "> &" << output_name << "_to_user" << ", ";
  }

  std::string inference_signature = ss.str();

  return inference_signature.substr(0, inference_signature.size() - 2);

}


std::string hypertea_func::copy_data_cpu(bool from_user) {

  std::stringstream ss;
  
  if (from_user) {
    for (auto const & input_name : input_names) {
      ss << "hypertea_copy(" << input_name + "_from_user" << ".size(), "
         << input_name + "_from_user" << ".data(), "
         << input_name + "_data" << ");" << std::endl;
    }
  } else {
    for (auto const & output_name : output_names) {
      ss << "hypertea_copy(" << output_name + "_to_user" << ".size(), "
         << output_name + "_data, "
         << output_name + "_to_user" << ".data()" << ");" << std::endl;
    }
  }
  
  return ss.str();

}



std::string hypertea_func::copy_data_gpu(bool from_user) {

  std::stringstream ss;
  
  if (from_user) {

    for (auto const & input_name : input_names) {
      ss << "OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, "
         << input_name + "_data" << ", "
         << "CL_TRUE, 0, "
         << input_name + "_from_user" << ".size() * sizeof(" << input_name + "_from_user[0]" << "), "
         << input_name + "_from_user" << ".data(), "
         << "0, NULL, NULL));" << std::endl;
    }
  } else {
    for (auto const & output_name : output_names) {
      ss << "OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, "
         << output_name + "_data" << ", "
         << "CL_TRUE, 0, "
         << output_name + "_to_user" << ".size() * sizeof(" << output_name + "_to_user[0]" << "), "
         << output_name + "_to_user" << ".data(), "
         << "0, NULL, NULL));" << std::endl;
    }
  }
  
  return ss.str();

}


std::string hypertea_func::hypertea_cpu(std::string net_name) {

  std::stringstream ss;

  ss << "#include \"hypertea/hypertea.hpp\"" << std::endl
     << std::endl;

  ss << "namespace hypertea {" << std::endl << std::endl;

  ss << "class " << net_name << std::endl << std::endl
     << "{" << std::endl
     << "public:" << std::endl;
  
  ss << net_name << "() { " << std::endl << std::endl
     << copy_weight_cpu(net_name) << std::endl << std::endl
     // << "FILE *f = fopen(" << "\"" << (net_name+".weight").c_str() << "\"" << ", \"rb\");" << std::endl
     // << "fread(all_weights, 1, weight_size, f);" << std::endl
     // << "fclose(f);" << std::endl
     << "}" << std::endl << std::endl << std::endl;


  ss << "~" << net_name << "() {" << std::endl << std::endl
     << "free(all_weights);" << std::endl << std::endl
     << "}" << std::endl;
  


  ss << "void inference( " << inference_signature() << ") { " << std::endl << std::endl << std::endl;

    ss << cpu_inoutputs_defs() << std::endl << std::endl;

    ss << copy_data_cpu(true) << std::endl << std::endl;

    ss << cpu_op_runs() << std::endl << std::endl;

    ss << copy_data_cpu(false) << std::endl << std::endl;

    ss << cpu_free_inoutputs() << std::endl << std::endl;

  ss << "}" << std::endl << std::endl << std::endl;


  ss << "private:" << std::endl << std::endl << std::endl;


  ss << cpu_param_defs() << std::endl << std::endl;

  ss << cpu_op_defs() << std::endl << std::endl;

  ss << "};" << std::endl;

  ss << "} //namespace hypertea" << std::endl;

  return ss.str();

}



std::string hypertea_func::hypertea_gpu(std::string net_name) {

  save_conv_file(net_name);

  std::stringstream ss;


  ss << "#include \"hypertea/hypertea.hpp\"" << std::endl
     << std::endl;

  ss << "namespace hypertea {" << std::endl << std::endl;

  ss << "class " << net_name << std::endl << std::endl
     << "{" << std::endl
     << "public:" << std::endl;
  
  ss << net_name << "() { " << std::endl << std::endl;

  ss << copy_weight_gpu(net_name) << std::endl << std::endl;

  ss << "OpenCLHandler::Get().build_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program);" << std::endl
     << "}" << std::endl << std::endl << std::endl;


  ss << "~" << net_name << "() {" << std::endl << std::endl
     << gpu_param_free() << std::endl
     << "}" << std::endl;
  


  ss << "void inference( " << inference_signature() << ") { " << std::endl << std::endl << std::endl;

    ss << gpu_inoutputs_defs() << std::endl << std::endl;

    ss << copy_data_gpu(true) << std::endl << std::endl;

    ss << gpu_op_runs() << std::endl << std::endl;

    ss << copy_data_gpu(false) << std::endl << std::endl;

    ss << gpu_free_inoutputs() << std::endl << std::endl;

  ss << "}" << std::endl << std::endl << std::endl;




  ss << "private:" << std::endl << std::endl << std::endl;

  ss << gpu_param_defs() << std::endl << std::endl;

  ss << gpu_op_defs() << std::endl << std::endl;

  ss << "};" << std::endl;



  ss << "} //namespace hypertea" << std::endl;
  return ss.str();
}

}