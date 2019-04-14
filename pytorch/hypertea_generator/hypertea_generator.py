import torch
from functools import partial, reduce
import numpy as np

from hypertea_generator.netinfo import NetInfo


class HyperteaGenerator(object):
    """docstring for HyperteaGenerator"""
    def __init__(self, net, input_tensor, precision):
        super(HyperteaGenerator, self).__init__()
        
        assert precision == 'float' or precision == 'half'

        self.precision = precision

        self.net_info_ = NetInfo(net, input_tensor, precision)

        self.declarations, self.parameters = self.net_info_.get_declare_with_param()

        

        

    
    def network_defination(self, inference_function, net_name):


        weight_size, weight_defs = self.para_defs_()

        self.write_opencl_code('work_space')


        return f"""
#include "hypertea/hypertea.hpp"

namespace hypertea {{

template <typename DeviceTensor>
class {net_name} {{

public:

    {net_name}(const std::string &param_file) {{ 

        compile_opencl_kernels(" ", " ");
        
        load_weight_to_tensor(param_file, {weight_size * self.sizeof_dtype_()}, param);

    }}

    {inference_function}

private:
    
    DeviceTensor param = DeviceTensor({weight_size});

    {self.new_line_joiner_(4).join(weight_defs)}
    {self.new_line_joiner_(4).join(self.op_defs_())}

}};


}} //namespace hypertea
        """
        

    def get_net_output(self):
        return self.net_info_.net_output 



    def op_defs_(self):
        op_declas = []
        for declaration in self.declarations:
            op_declas.append('{0}<DeviceTensor> {1} = {0}<DeviceTensor> ({2});'.format(
                declaration['type'], declaration['op_name'], declaration['signature'])
            )
        return op_declas


    def para_defs_(self):
        weight_declares, current_pos = [], 0
        with open('work_space/pytorch_weight', 'wb') as f:
            for parameter in self.parameters:
                self.write_parameter_(parameter[1], f)
                weight_declares.append(' DeviceTensor {0} = param.sub_view({1}, {2});'.format(parameter[0], current_pos, parameter[1].nelement()))
                current_pos += parameter[1].nelement()
        return current_pos, weight_declares


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



    def write_opencl_code(self, dir_path):
        for file_path, opencl_code in self.net_info_.get_opencl_kernels():
            with open('{}/{}'.format(dir_path, file_path), 'w') as f:
                f.write(opencl_code)


    def new_line_joiner_(self, space):
        return '\n' + ' ' * space



    