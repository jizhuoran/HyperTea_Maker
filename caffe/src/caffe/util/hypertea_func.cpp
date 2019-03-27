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


void hypertea_func::append_op_inference(FlySignature signature) {
  op_inference.push_back(signature);
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



std::string map_name(std::string name, std::map<std::string, std::string> map) {
  return map.find(name) == map.end() ? name : map[name];
}


void print_refer_map(std::map<std::string, int> refer_map) {
  for(auto const& op_para:refer_map) {
    std::cout << op_para.first << " is " << op_para.second << std::endl;
  }
  std::cout << "  -----  " << std::endl;


}


/*
std::string hypertea_func::cpu_op_runs() { 
  
  std::stringstream ss;
  
  // for(auto const& op_para:op_run) {
  //   ss << op_para << std::endl;

  // }

  std::map<std::string, std::string> split_tracker;

  for(auto const& op_para:op_inference) {
    if (op_para.op_type_ == "Split") {
      std::string input_name = op_para.inputs_name_[0];
      for(auto const& output_name:op_para.outputs_name_) {
        split_tracker[output_name] = input_name;
      }
    }
  }


  for(auto & op_para:op_inference) {
    
    for(auto & input_name:op_para.inputs_name_) {
      input_name = desplit(input_name, split_tracker);
    }

    for(auto & output_name:op_para.outputs_name_) {
      output_name = desplit(output_name, split_tracker);
    }

  }



  std::map<std::string, std::vector<int> > name_reference_count;

  for(auto const& op_para:op_inference) {
    
    for(auto const& input_name:op_para.inputs_name_) {
      name_reference_count[input_name].push_back(-1);
    }

    for(auto const& output_name:op_para.outputs_name_) {
      name_reference_count[output_name].push_back(1);
    }

  }


  std::map<std::string, int > name_max_reference;

  for(auto const& name_reference:name_reference_count) {
    

    int max_reference = -1;
    int sum = 0;
    for(auto const& refer:name_reference.second) {
      sum += refer;
      max_reference = std::max(sum, max_reference);
    }

    name_max_reference[name_reference.first] = max_reference;

  }



  std::map<std::string, std::string> name_map;
  std::set<std::string> defined_tensor;

  for(auto const& op_para:op_inference) {
    
    if (op_para.op_type_ == "Split") {
      continue;
    } else if (op_para.op_type_ == "Eltwise") {

      // std::string input_name = op_para.inputs_name_[0];
      // std::string input_name1 = op_para.inputs_name_[1];
      // std::string output_name = op_para.outputs_name_[0];

      std::string input_name = desplit(op_para.inputs_name_[0], name_map);
      std::string input_name1 = desplit(op_para.inputs_name_[1], name_map);
      std::string output_name = desplit(op_para.outputs_name_[0], name_map);

      name_max_reference[input_name] -= 1;
      name_max_reference[input_name1] -= 1;
      name_max_reference[output_name] += 1;


      if (name_max_reference[input_name1] == 1) {
          name_map[output_name] = input_name1;
          // name_max_reference[input_name1] += 1;
          if(input_name1 == output_name) {
            name_max_reference[input_name] = name_max_reference[output_name];
          } else {
            name_max_reference[input_name] = 0;
          }
      }

      if (name_max_reference[input_name] == 1) {
          name_map[output_name] = input_name;
          // name_max_reference[input_name] += 1;

          if(input_name == output_name) {
            name_max_reference[input_name1] = name_max_reference[output_name];
          } else {
            name_max_reference[input_name1] = 0;
          }
      }


      if (name_max_reference[input_name] == 1 || name_max_reference[input_name1] == 1) {
        name_max_reference[output_name] = 0;
      }



      output_name = desplit(output_name, name_map);

      print_refer_map(name_max_reference);




      if(input_name == output_name) {
        if (defined_tensor.find(output_name) == defined_tensor.end()) {
          defined_tensor.insert(output_name);
          output_name = "auto " + output_name;
        }
        ss << output_name << "_data += " << input_name1 << "_data;" << std::endl;
      } else if (input_name1 == output_name){
        if (defined_tensor.find(output_name) == defined_tensor.end()) {
          defined_tensor.insert(output_name);
          output_name = "auto " + output_name;
        }
        ss << output_name << "_data += " << input_name << "_data;" << std::endl;

      } else {
        if (defined_tensor.find(output_name) == defined_tensor.end()) {
          defined_tensor.insert(output_name);
          output_name = "auto " + output_name;
        }
        ss << output_name << "_data = " << input_name << "_data + "<< input_name1 << "_data;" << std::endl;
      }


      std::cout << output_name << "_data = " << input_name << "_data + "<< input_name1 << "_data;" << std::endl;

    } else {

      std::string input_name = desplit(op_para.inputs_name_[0], name_map);
      std::string output_name = desplit(op_para.outputs_name_[0], name_map);

      name_max_reference[input_name] -= 1;
      name_max_reference[output_name] += 1;


      if (name_max_reference[input_name] == 1) {
          name_map[output_name] = input_name;
          name_max_reference[input_name] = name_max_reference[output_name];
          name_max_reference[output_name] = 0;

      }

      output_name = desplit(output_name, name_map);

      print_refer_map(name_max_reference);

      if (defined_tensor.find(output_name) == defined_tensor.end()) {
        defined_tensor.insert(output_name);
        output_name = "auto " + output_name;
      }

      ss << output_name << "_data = " << op_para.op_name_ << "(" << input_name << "_data);" << std::endl;
      std::cout << output_name << "_data = " << op_para.op_name_ << "(" << input_name << "_data);" << std::endl;

    }

  }




  return ss.str();

}
*/


void hypertea_func::map_split_name_back() {


//In Caffe, the blob used by more than 1 layer will be splited into several new blobs
//with automatic generated name. This function get the map generated_name->origin_name

    std::map<std::string, std::string> map;

    for(auto const& op_para:op_inference) {
      if (op_para.op_type_ == "Split") {
        std::string input_name = op_para.inputs_name_[0];
        for(auto const& output_name:op_para.outputs_name_) {
          map[output_name] = input_name;
        }
      }
    }


    for(auto & op_para:op_inference) {
      for(auto & input_name:op_para.inputs_name_) {
        input_name = map_name(input_name, map);
      }
      for(auto & output_name:op_para.outputs_name_) {
        output_name = map_name(output_name, map);
      }
    }
  
}




int find_max_sublist (std::vector<int> l) {
    
    int max_ending_here = l[0];
    int max_so_far = l[0];

    for (auto x = std::next(l.begin()); x != l.end(); ++x) {
      max_ending_here = std::max(*x, max_ending_here + *x);
      max_so_far = std::max(max_so_far, max_ending_here);
    }

    return max_so_far;
}



std::map<std::string, int> hypertea_func::get_names_max_refer_time() {

  std::map<std::string, std::vector<int> > name_refer_count;

  for(auto const& op_para:op_inference) {
    
    for(auto const& input_name:op_para.inputs_name_) {
      name_refer_count[input_name].push_back(-1);
    }

    for(auto const& output_name:op_para.outputs_name_) {
      name_refer_count[output_name].push_back(1);
    }

  }

  std::map<std::string, int > name_max_refer;

  for(auto const& name_reference:name_refer_count) {
    name_max_refer[name_reference.first] = std::abs(find_max_sublist(name_reference.second));
  }

  return name_max_refer;

}



std::string append_declare_if_need(std::string name, 
                                   std::set<std::string> &defined_name,
                                   std::map<std::string, int> &refer_count) {
  
  if (defined_name.find(name) == defined_name.end()) {
    defined_name.insert(name);
    refer_count[name] -= 1;
    return "auto " + name;
  }

  return name;
}





std::string hypertea_func::cpu_op_runs() { 
  
  std::stringstream ss;
  
  map_split_name_back();

  std::map<std::string, std::vector<int> > name_reference_count;


  std::map<std::string, int > name_max_reference = get_names_max_refer_time();

  std::set<std::string> defined_tensor(input_names.begin(), input_names.end());


  // std::map<std::string, std::string> name_map;


  for(auto const& op_para:op_inference) {
    
    if (op_para.op_type_ == "Eltwise") {


      std::string lhs_input = map_name(op_para.inputs_name_[0], name_map);
      std::string rhs_input = map_name(op_para.inputs_name_[1], name_map);
      std::string output_name = map_name(op_para.outputs_name_[0], name_map);

      name_max_reference[lhs_input] -= 1;
      name_max_reference[rhs_input] -= 1;
      name_max_reference[output_name] += 1;


      bool lhf_free = name_max_reference[lhs_input] == 0;
      bool rhf_free = name_max_reference[rhs_input] == 0;


      if (lhf_free) {
          name_map[output_name] = lhs_input;
          name_max_reference[lhs_input] = name_max_reference[output_name] - 1;
          name_max_reference[output_name] = 0;

          if(rhf_free) { 
            name_max_reference[rhs_input] = 0;
          }

      } else if (rhf_free) {
          name_map[output_name] = rhs_input;
          name_max_reference[rhs_input] = name_max_reference[output_name] - 1;
          name_max_reference[output_name] = 0;

      } 

      output_name = map_name(output_name, name_map);

      print_refer_map(name_max_reference);


      output_name = append_declare_if_need(output_name, defined_tensor, name_max_reference);

      if(lhs_input == output_name) {
        ss << output_name << "_data += " << rhs_input << "_data;" << std::endl;
      } else if (rhs_input == output_name){
        ss << output_name << "_data += " << lhs_input << "_data;" << std::endl;
      } else {
        ss << output_name << "_data = " << lhs_input << "_data + "<< rhs_input << "_data;" << std::endl;
      }


      std::cout << output_name << "_data = " << lhs_input << "_data + "<< rhs_input << "_data;" << std::endl;

    } else if (op_para.op_type_ != "Split") {

      std::string input_name = map_name(op_para.inputs_name_[0], name_map);
      std::string output_name = map_name(op_para.outputs_name_[0], name_map);

      name_max_reference[input_name] -= 1;
      name_max_reference[output_name] += 1;


      if (name_max_reference[input_name] == 0) {
          name_map[output_name] = input_name;
          name_max_reference[input_name] = name_max_reference[output_name] - 1;
          name_max_reference[output_name] = 0;
      }


      output_name = map_name(output_name, name_map);

      output_name = append_declare_if_need(output_name, defined_tensor, name_max_reference);

      print_refer_map(name_max_reference);

      ss << output_name << "_data = " << op_para.op_name_ << "(" << input_name << "_data);" << std::endl;
      std::cout << output_name << "_data = " << op_para.op_name_ << "(" << input_name << "_data);" << std::endl;

    }

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
      
      ss << "Tensor<" << this->dtype_ << "> " << name << "_data"
         << "(" << info.length_ << ");" << std::endl;

      // ss << this->dtype_ << "* " << name << "_data = ("
      //    << this->dtype_ << " *)malloc(" << this->dtype_size_ * info.length_ << ");" << std::endl;

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
      ss << "TensorCPU<" << this->dtype_ << "> " << input_name + "_data"
         << "(" << input_name + "_from_user" << ");" << std::endl;
    }
  } else {
    for (auto const & output_name : output_names) {
      ss << "hypertea_copy(" << output_name + "_to_user" << ".size(), "
         << map_name(output_name, name_map) + "_data.immutable_data(), "
         << output_name + "_to_user" << ".data()" << ");" << std::endl;
    }
  }
  
  return ss.str();

}



std::string hypertea_func::copy_data_gpu(bool from_user) {

  std::stringstream ss;
  
  if (from_user) {

    for (auto const & input_name : input_names) {
      ss << "TensorGPU<" << this->dtype_ << "> " << input_name + "_data"
         << "(" << input_name + "_from_user" << ");" << std::endl;
    }
  } else {
    for (auto const & output_name : output_names) {
      ss << "OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, "
         << map_name(output_name, name_map) + "_data.immutable_data()" << ", "
         << "CL_TRUE, 0, "
         << output_name + "_to_user" << ".size() * sizeof(" << output_name + "_to_user[0]" << "), "
         << output_name + "_to_user" << ".data(), "
         << "0, NULL, NULL));" << std::endl;
    }
  }
  
  return ss.str();

}


std::string hypertea_func::hypertea_cpu(std::string net_name) {

  name_map.clear();


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

    // ss << cpu_inoutputs_defs() << std::endl << std::endl;

    ss << copy_data_cpu(true) << std::endl << std::endl;

    ss << cpu_op_runs() << std::endl << std::endl;

    ss << copy_data_cpu(false) << std::endl << std::endl;

    // ss << cpu_free_inoutputs() << std::endl << std::endl;

  ss << "}" << std::endl << std::endl << std::endl;


  ss << "private:" << std::endl << std::endl << std::endl;


  ss << cpu_param_defs() << std::endl << std::endl;

  ss << cpu_op_defs() << std::endl << std::endl;

  ss << "};" << std::endl;

  ss << "} //namespace hypertea" << std::endl;

  return ss.str();

}



std::string hypertea_func::hypertea_gpu(std::string net_name) {

  name_map.clear();

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

    // ss << gpu_inoutputs_defs() << std::endl << std::endl;

    ss << copy_data_gpu(true) << std::endl << std::endl;

    ss << gpu_op_runs() << std::endl << std::endl;

    ss << copy_data_gpu(false) << std::endl << std::endl;

    // ss << gpu_free_inoutputs() << std::endl << std::endl;

  ss << "}" << std::endl << std::endl << std::endl;




  ss << "private:" << std::endl << std::endl << std::endl;

  ss << gpu_param_defs() << std::endl << std::endl;

  ss << gpu_op_defs() << std::endl << std::endl;

  ss << "};" << std::endl;



  ss << "} //namespace hypertea" << std::endl;
  return ss.str();
}

}
