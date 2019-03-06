import torch
from functools import partial, reduce
import numpy as np

from libdnn import Libdnn

class HyperteaGenerator(object):
    """docstring for HyperteaGenerator"""
    def __init__(self, net):
        super(HyperteaGenerator, self).__init__()
        self.net = net
        
        self.parameters = []
        self.declarations = []

        self.libdnn_conv_str = ''

        self.hooker_map = {
            torch.nn.modules.conv.Conv2d:self.conv_hooker_,
            torch.nn.modules.conv.ConvTranspose2d:self.conv_hooker_,
            torch.nn.modules.batchnorm.BatchNorm2d:self.batch_norm_hooker_,
            torch.nn.modules.activation.ELU:self.elu_hooker_,
            torch.nn.modules.activation.ReLU:self.relu_hooker_,
            torch.nn.modules.activation.Tanh:self.tanh_hooker_
        }

        self.attach_hooker_()

    
    def hypertea_gpu(self, inference_code):

        weight_size, weight_defs, weight_copy, weight_free = self.para_defs_gpu_()

        return f"""
#include "hypertea/hypertea.hpp"

namespace hypertea {{

class new_net {{
public:

    new_net() {{

        int weight_size = {weight_size};
        unsigned char* all_weights = (unsigned char*) malloc(weight_size);

        FILE *f = fopen("pytorch_weight", "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) {{ 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }}
        fclose(f);

        {self.new_line_joiner_(8).join(weight_copy)}

        free(all_weights);
        OpenCLHandler::Get().build_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program);

    }}


    ~new_net() {{
        {self.new_line_joiner_(8).join(weight_free)}
    }}

    {inference_code}

private:

    {self.new_line_joiner_(4).join(weight_defs)}


    {self.new_line_joiner_(4).join(self.op_defs_gpu_())}

}};
}} //namespace hypertea
        """
        

    

    def hypertea_cpu(self, inference_code):

        weight_size, weight_defs = self.para_defs_cpu_()

        hypertea_code = """
#include "hypertea/hypertea.hpp"

namespace hypertea {{

class new_net {{
public:

    new_net() {{

        FILE *f = fopen("pytorch_weight", "rb");
        size_t read_size = fread(all_weights, 1, weight_size, f);
        if (read_size != weight_size) {{ 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }}
        fclose(f);
    }}


    ~new_net() {{
        free(all_weights);
    }}

    {0}

private:
    int weight_size = {1};
    unsigned char* all_weights = (unsigned char*) malloc(weight_size);
{2}


{3}

}};
}} //namespace hypertea
        """
        return hypertea_code.format(inference_code, weight_size, weight_defs, self.op_defs_cpu_())



    def write_conv_file_(self):


        header = '''std::string conv_opencl_funcs = R"(#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
'''
        with open('/home/zrji/hypertea/tools/conv_opencl.hpp', 'w') as f:
            f.write(header)
            f.write(self.libdnn_conv_str)
            f.write(')";')



    def forward_to_collect_info(self, input_tensor):

        self.net(input_tensor)


    def attach_hooker_(self):
        for name, module in self.net.named_modules():
            if type(module) in self.hooker_map:
                module.register_forward_hook(partial(self.hooker_map[type(module)], op_name = name))


    def op_defs_cpu_(self):
        op_declas = []
        for declaration in self.declarations:
            op_declas.append('{}_CPU<float> {} = {}_CPU<float> ({});'.format(
                declaration['type'], declaration['op_name'],
                declaration['type'], declaration['cpu_signature'])
            )
        return '\n'.join(op_declas)


    def op_defs_gpu_(self):
        op_declas = []
        for declaration in self.declarations:
            op_declas.append('{}_GPU<float> {} = {}_GPU<float> ({});'.format(
                declaration['type'], declaration['op_name'],
                declaration['type'], declaration['gpu_signature'])
            )
        return op_declas


    def para_defs_cpu_(self):
        weight_declares, current_pos = [], 0
        with open('pytorch_weight', 'wb') as f:
            for parameter in self.parameters:
                f.write(np.array(parameter[1].tolist(), np.float32).tobytes())
                weight_declares.append('float* {} = reinterpret_cast<float*>(all_weights + {});'.format(parameter[0], current_pos))
                current_pos += parameter[1].nelement() * 4

        return current_pos, '\n'.join(weight_declares)


    def para_defs_gpu_(self):
        weight_declares, weight_copy, weight_free, current_pos = [], [], [], 0
        with open('pytorch_weight', 'wb') as f:
            for parameter in self.parameters:
                f.write(np.array(parameter[1].tolist(), np.float32).tobytes())
                weight_copy.append('OPENCL_CHECK(clEnqueueWriteBuffer(OpenCLHandler::Get().commandQueue, {}, CL_TRUE, 0, {}, all_weights + {}, 0, NULL, NULL));'.format(parameter[0], parameter[1].nelement() * 4, current_pos))
                weight_free.append('OPENCL_CHECK(clReleaseMemObject({}));'.format(parameter[0]))
                weight_declares.append('cl_mem {} = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_ONLY, {}, NULL, NULL);'.format(parameter[0], parameter[1].nelement() * 4))
                current_pos += parameter[1].nelement() * 4

        return current_pos, weight_declares, weight_copy, weight_free



    def batch_norm_hooker_(self, module, input, output, op_name):


        top_shape = input[0].nelement()
        num, channels = input[0].shape[:2]
        eps = module.eps
        scale_factor = 1
        use_global_stats = self.bool2str_(module.track_running_stats)
        bias_name, bias = op_name+'_bias', module.bias
        weight_name, weight = op_name+'_weight', module.weight


        if module.track_running_stats:
            mean_name, var_name = op_name+'_mean', op_name+'_var'
            self.parameters.append((mean_name, module.running_mean))
            self.parameters.append((var_name,  module.running_var))
        else:
            mean_name = var_name = 'NULL'


        if module.affine:
            weight_name, bias_name = op_name+'_weight', op_name+'_bias'
            self.parameters.append((weight_name, module.weight))
            self.parameters.append((bias_name,  module.bias))
        else:
            weight_name = bias_name = 'NULL'


        cpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                    top_shape, num, channels, eps, scale_factor, 
                    use_global_stats, mean_name, var_name,
                    weight_name, bias_name)
        gpu_signature = cpu_signature
        self.declarations.append({'type':'BatchNormOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})



    def conv_hooker_(self, module, input_data, output_data, op_name):


        if module.bias is not None:
            bias_name, bias = op_name+'_bias', module.bias
            self.parameters.append((bias_name, bias))
        else:
            bias_name = 'NULL'


        weight_name, weight = op_name+'_weight', module.weight
        self.parameters.append((weight_name, weight))

        groups = module.groups

        kernel_shape = list(module.kernel_size)
        stride = list(module.stride)
        padding = list(module.padding)
        dilation = list(module.dilation)


        is_1x1 = all(list(map(lambda x: x==1, kernel_shape + stride)) + list(map(lambda x: x==0, padding)))

        input_shape = list(input_data[0].shape)
        output_shape = [len(output_data)] + list(output_data[0].shape)

        force_nd_conv = False

        conv_type = 'DeconvolutionOp' if module.transposed else 'ConvolutionOp'

        cpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                    weight_name, bias_name, groups, self.bool2str_(is_1x1), 
                    self.list2vecstr_(kernel_shape), self.list2vecstr_(stride), 
                    self.list2vecstr_(padding), self.list2vecstr_(dilation),
                    self.list2vecstr_(input_shape), self.list2vecstr_(output_shape),
                    self.bool2str_(force_nd_conv))


        libdnn = Libdnn(op_name+'_forward', groups,
                                conv_type, module.bias is not None, 
                                input_shape, output_shape, 
                                kernel_shape, padding, stride, dilation)

        self.libdnn_conv_str += libdnn.generate_libdnn_code()


        gpu_signature = '"{}_forward", {}, {}, {}, {}, {}'.format(
            op_name, self.prod_(output_shape), 
            weight_name, bias_name,
            self.list2vecstr_(libdnn.local_shape()),
            self.list2vecstr_(libdnn.global_shape()))




        self.declarations.append({'type':conv_type, 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})



    def relu_hooker_(self, module, input_data, output_data, op_name):
        cpu_signature = ' 0, {} '.format(self.bool2inplace_str_(module.inplace))
        gpu_signature = cpu_signature
        self.declarations.append({'type':'ReLUOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


    def elu_hooker_(self, module, input_data, output_data, op_name):
        cpu_signature = ' 1, {} '.format(self.bool2inplace_str_(module.inplace))
        gpu_signature = cpu_signature
        self.declarations.append({'type':'ELUOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


    def tanh_hooker_(self, module, input_data, output_data, op_name):
        cpu_signature = ' NOT_IN_PLACE '
        gpu_signature = cpu_signature
        self.declarations.append({'type':'TanHOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})

    def prod_(self, l):
        return reduce(lambda x, y: x*y, l)

    def list2vecstr_(self, l):
        return 'std::vector<int> {'  + ','.join(map(str, l)) + '}'

    def bool2str_(self, x):
        return 'true' if x else 'false'

    def bool2inplace_str_(self, x):
        return 'IN_PLACE' if x else 'NOT_IN_PLACE'

    def new_line_joiner_(self, space):
        return '\n' + ' ' * space

    