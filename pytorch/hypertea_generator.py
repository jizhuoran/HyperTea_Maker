import torch
from functools import partial, reduce
import numpy as np


from netinfo import NetInfo

class HyperteaGenerator(object):
    """docstring for HyperteaGenerator"""
    def __init__(self, net, input_tensor, precision):
        super(HyperteaGenerator, self).__init__()
        
        if not (precision == 'float' or precision == 'half'):
            print("unrecognized precision, we only support float and half")
            exit(1)
        self.precision = precision

        self.net_info_ = NetInfo(net, input_tensor, precision)
        self.declarations, self.parameters = self.net_info_.get_declare_with_param()


        self.conv_opencl_kernel, self.bn_opencl_kernel = self.net_info_.get_opencl_kernels()
        

        

    
    def hypertea_gpu(self, inference_code, net_name, android_code = False):

    
        static_declare = 'static {0} *{0}_jni_;'.format(net_name) if android_code else ' '
        static_init = '{0} *{0}::{0}_jni_ = nullptr;'.format(net_name) if android_code else ' '


        weight_size, weight_defs, weight_copy, weight_free = self.para_defs_gpu_()

        self.write_opencl_code_('temp_conv.cl', self.conv_opencl_kernel)
        self.write_opencl_code_('temp_bn.cl', self.bn_opencl_kernel)


        return f"""
#include "hypertea/hypertea.hpp"
#include "bn_opencl.hpp"
#include "conv_opencl.hpp"

namespace hypertea {{

class {net_name} {{
public:

    {self.android_helper_code(net_name) if android_code else ' '}

    {net_name}(const std::string &param_file) {{

        int weight_size = {weight_size};
        unsigned char* all_weights = (unsigned char*) malloc(weight_size);

        FILE *f = fopen(param_file.c_str(), "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) {{ 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }}
        fclose(f);

        {self.new_line_joiner_(8).join(weight_copy)}

        free(all_weights);
        OpenCLHandler::Get().build_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program);
        OpenCLHandler::Get().build_opencl_program(bn_opencl_funcs, OpenCLHandler::Get().bn_program);

    }}


    ~{net_name}() {{
        {self.new_line_joiner_(8).join(weight_free)}
    }}

    {inference_code}

private:

    {static_declare}

    {self.new_line_joiner_(4).join(weight_defs)}


    {self.new_line_joiner_(4).join(self.op_defs_gpu_())}

}};

{static_init}

}} //namespace hypertea
        """
        

    

    def hypertea_cpu(self, inference_code, net_name):

        weight_size, weight_defs = self.para_defs_cpu_()

        return f"""
#include "hypertea/hypertea.hpp"

namespace hypertea {{

class {net_name} {{
public:

    {net_name}() {{

        FILE *f = fopen("pytorch_weight", "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) {{ 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }}
        fclose(f);
    }}


    ~{net_name}() {{
        free(all_weights);
    }}

    {inference_code}

private:
    int weight_size = {weight_size};
    unsigned char* all_weights = (unsigned char*) malloc(weight_size);
{weight_defs}


{self.op_defs_cpu_()}

}};
}} //namespace hypertea
        """





    def android_helper_code(self, net_name):
        return f'''
    
    static {net_name} *get() {{
        return {net_name}_jni_;
    }}


    static {net_name} *get(const std::string &param_file) {{
            {net_name}_jni_ = new {net_name}(param_file);
            return {net_name}_jni_;
    }} 

    '''
    


    def get_net_output(self):
        return self.net_info_.net_output 




    def op_defs_cpu_(self):
        op_declas = []
        for declaration in self.declarations:
            op_declas.append('{0}_CPU<{4}> {1} = {2}_CPU<{4}> ({3});'.format(
                declaration['type'], declaration['op_name'],
                declaration['type'], declaration['cpu_signature'],
                self.precision)
            )
        return '\n'.join(op_declas)


    def op_defs_gpu_(self):
        op_declas = []
        for declaration in self.declarations:
            op_declas.append('{0}_GPU<{4}> {1} = {2}_GPU<{4}> ({3});'.format(
                declaration['type'], declaration['op_name'],
                declaration['type'], declaration['gpu_signature'],
                self.precision)
            )
        return op_declas


    def para_defs_cpu_(self):
        weight_declares, current_pos = [], 0
        with open('pytorch_weight', 'wb') as f:
            for parameter in self.parameters:
                self.write_parameter_(parameter[1], f)
                # f.write(np.array(parameter[1].tolist(), np.float32).tobytes())
                weight_declares.append('{0}* {1} = reinterpret_cast<{0}*>(all_weights + {2});'.format(self.precision, parameter[0], current_pos))
                current_pos += parameter[1].nelement() * self.sizeof_dtype_()

        return current_pos, '\n'.join(weight_declares)


    def para_defs_gpu_(self):
        weight_declares, weight_copy, weight_free, current_pos = [], [], [], 0
        with open('pytorch_weight', 'wb') as f:
            for parameter in self.parameters:
                self.write_parameter_(parameter[1], f)
                # f.write(np.array(parameter[1].tolist(), np.float32).tobytes())
                weight_copy.append('OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, {}, CL_TRUE, 0, {}, all_weights + {}, 0, NULL, NULL));'.format(parameter[0], parameter[1].nelement() * self.sizeof_dtype_(), current_pos))
                weight_free.append('OPENCL_CHECK(clReleaseMemObject({}));'.format(parameter[0]))
                weight_declares.append('cl_mem {} = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, {}, NULL, NULL);'.format(parameter[0], parameter[1].nelement() * self.sizeof_dtype_()))
                current_pos += parameter[1].nelement() * self.sizeof_dtype_()

        return current_pos, weight_declares, weight_copy, weight_free


    def sizeof_dtype_(self):
        if self.precision == 'float':
            return 4
        elif self.precision == 'half':
            return 2


    def write_parameter_(self, parameters, f):
        if self.precision == 'float':
            f.write(np.array(parameters.tolist(), np.float32).tobytes())
        elif self.precision == 'half':
            f.write(np.array(parameters.tolist(), np.float16).tobytes())



    def write_opencl_code_(self, file_name, opencl_code):
        with open(file_name, 'w') as f:
            f.write(opencl_code)


    def new_line_joiner_(self, space):
        return '\n' + ' ' * space



    