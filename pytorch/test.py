# -*- coding: utf-8 -*-
import torch
from net import LSTMClassifier as Model
from hypertea_generator import HyperteaGenerator



import cv2
import numpy as np
im = np.array(cv2.imread("HKU.jpg"))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = im.transpose(2,0,1)


model = Model().train()
# model.load_state_dict(torch.load('styte_net'))

genetator = HyperteaGenerator(model)
genetator.forward_to_collect_info(torch.tensor(im.reshape(1,3,512,512), dtype = torch.float))

cpu_code = '''
    
    void inference( std::vector<float> &data_from_user, std::vector<float> &data_to_user) {
        
        TensorCPU<float> data(data_from_user);

        auto temp = bn1(elu1(conv1(data)));
        temp = bn2(elu2(conv2(temp)));
        temp = bn3(elu3(conv3(temp)));


        temp += res1_bn2(res1_conv2(res1_bn1(res1_relu1(res1_bn1(res1_conv1(temp))))));
        temp += res2_bn2(res2_conv2(res2_bn1(res2_relu1(res2_bn1(res2_conv1(temp))))));
        temp += res3_bn2(res3_conv2(res3_bn1(res3_relu1(res3_bn1(res3_conv1(temp))))));
        temp += res4_bn2(res4_conv2(res4_bn1(res4_relu1(res4_bn1(res4_conv1(temp))))));
        temp += res5_bn2(res5_conv2(res5_bn1(res5_relu1(res5_bn1(res5_conv1(temp))))));


        temp = de_bn1(de_elu1(deconv1(temp)));
        temp = de_bn2(de_elu2(deconv2(temp)));
        temp = de_tanh3(deconv3(temp));

        temp = (temp + 1) * 127.5;

        hypertea_copy(data_to_user.size(), temp.data(), data_to_user.data());

    }
'''


gpu_code = '''
    
    void inference( std::vector<float> &data_from_user, std::vector<float> &data_to_user) {
        
        TensorGPU<float> data(data_from_user);

        auto temp = bn1(elu1(conv1(data)));
        temp = bn2(elu2(conv2(temp)));
        temp = bn3(elu3(conv3(temp)));


        temp += res1_bn2(res1_conv2(res1_relu1(res1_bn1(res1_conv1(temp)))));
        temp += res2_bn2(res2_conv2(res2_relu1(res2_bn1(res2_conv1(temp)))));
        temp += res3_bn2(res3_conv2(res3_relu1(res3_bn1(res3_conv1(temp)))));
        temp += res4_bn2(res4_conv2(res4_relu1(res4_bn1(res4_conv1(temp)))));
        temp += res5_bn2(res5_conv2(res5_relu1(res5_bn1(res5_conv1(temp)))));
        

        temp = de_bn1(de_elu1(deconv1(temp)));
        temp = de_bn2(de_elu2(deconv2(temp)));
        temp = de_tanh3(deconv3(temp));

        temp = (temp + 1) * 127.5;

        OPENCL_CHECK(clEnqueueReadBuffer(OpenCLHandler::Get().commandQueue, temp.immutable_data(), CL_TRUE, 0, data_to_user.size() * sizeof(data_to_user[0]), data_to_user.data(), 0, NULL, NULL));

    }
'''

print(genetator.hypertea_gpu(gpu_code))
















# exit(0)

# a = nn.Module()



# parameters = []
# declarations = []


# def batch_norm_hooker(module, input, output, op_name):


#     top_shape = input[0].nelement()
#     num, channels = input[0].shape[:2]
#     eps = module.eps
#     scale_factor = 1
#     use_global_stats = bool2str(module.track_running_stats)
#     bias_name, bias = op_name+'_bias', module.bias
#     weight_name, weight = op_name+'_weight', module.weight


#     if module.track_running_stats:
#         mean_name, var_name = op_name+'_mean', op_name+'_var'
#         parameters.append((mean_name, module.running_mean))
#         parameters.append((var_name,  module.running_var))
#     else:
#         mean_name = var_name = 'NULL'


#     if module.affine:
#         weight_name, bias_name = op_name+'_weight', op_name+'_bias'
#         parameters.append((weight_name, module.weight))
#         parameters.append((bias_name,  module.bias))
#     else:
#         weight_name = bias_name = 'NULL'


#     cpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
#                 top_shape, num, channels, eps, scale_factor, 
#                 use_global_stats, mean_name, var_name,
#                 weight_name, bias_name)
#     gpu_signature = ' '
#     declarations.append({'type':'BatchNormOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})



# def conv_hooker(module, input_data, output_data, op_name):


#     if module.bias is not None:
#         bias_name, bias = op_name+'_bias', module.bias
#         parameters.append((bias_name, bias))
#     else:
#         bias_name = 'NULL'


#     weight_name, weight = op_name+'_weight', module.weight
#     parameters.append((weight_name, weight))

#     groups = module.groups

#     kernel_shape = list(module.kernel_size)
#     stride = list(module.stride)
#     padding = list(module.padding)
#     dilation = list(module.dilation)


#     is_1x1 = all(list(map(lambda x: x==1, kernel_shape + stride)) + list(map(lambda x: x==0, padding)))

#     input_shape = list(input_data[0].shape)
#     output_shape = [len(output_data)] + list(output_data[0].shape)

#     force_nd_conv = False

#     conv_type = 'DeconvolutionOp' if module.transposed else 'ConvolutionOp'

#     cpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
#                 weight_name, bias_name, groups, bool2str(is_1x1), 
#                 list2vecstr(kernel_shape), list2vecstr(stride), 
#                 list2vecstr(padding), list2vecstr(dilation),
#                 list2vecstr(input_shape), list2vecstr(output_shape),
#                 bool2str(force_nd_conv))
#     gpu_signature = ' '
#     declarations.append({'type':conv_type, 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})



# def relu_hooker(module, input_data, output_data, op_name):
#     cpu_signature = ' 0, {} '.format(bool2inplace_str(module.inplace))
#     gpu_signature = ' '
#     declarations.append({'type':'ReLUOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


# def elu_hooker(module, input_data, output_data, op_name):
#     cpu_signature = ' 1, {} '.format(bool2inplace_str(module.inplace))
#     gpu_signature = ' '
#     declarations.append({'type':'ELUOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


# def tanh_hooker(module, input_data, output_data, op_name):
#     cpu_signature = ' NOT_IN_PLACE '
#     gpu_signature = ' '
#     declarations.append({'type':'TanHOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})




# hooker_map = {
#     torch.nn.modules.conv.Conv2d:conv_hooker,
#     torch.nn.modules.conv.ConvTranspose2d:conv_hooker,
#     torch.nn.modules.batchnorm.BatchNorm2d:batch_norm_hooker,
#     torch.nn.modules.activation.ELU:elu_hooker,
#     torch.nn.modules.activation.ReLU:relu_hooker,
#     torch.nn.modules.activation.Tanh:tanh_hooker
# }






# for name, m in model.named_modules():
#     if type(m) in hooker_map:
#         m.register_forward_hook(partial(hooker_map[type(m)], op_name = name))




# def op_defs():

#     op_declas = []

#     for declaration in declarations:

#         op_declas.append('{}_CPU<float> {} = {}_CPU<float> ({});'.format(
#             declaration['type'], declaration['op_name'],
#             declaration['type'], declaration['cpu_signature'])
#         )

#     return '\n'.join(op_declas)


# def para_defs():

#     weight_declares = []
#     current_pos = 0

#     with open('pytorch_weight', 'wb') as f:
#         for parameter in parameters:
#             f.write(np.array(parameter[1].tolist(), np.float32).tobytes())
#             weight_declares.append('float* {} = reinterpret_cast<float*>(all_weights + {});'.format(parameter[0], current_pos))
#             current_pos += parameter[1].nelement() * 4

#     return current_pos, '\n'.join(weight_declares)


# import cv2
# im = np.array(cv2.imread("HKU.jpg"))

# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# im = im.transpose(2,0,1)


# output = model(torch.tensor(im, dtype = torch.float).reshape(1, 3, 512, 512))


# # print(output.reshape(3, 512, 512).detach().numpy())

# img = output.reshape(3, 512, 512).detach().numpy().transpose(1,2,0)

# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# cv2.imwrite('output.jpg', img)







# #        temp = (temp + 1) * 127.5;


# def hypertea_cpu(inference_code):

#     weight_size, weight_defs = para_defs()

#     a = """
# #include "hypertea/hypertea.hpp"

# namespace hypertea {{

# class new_net {{
# public:

#     new_net() {{

#         FILE *f = fopen("pytorch_weight", "rb");
#         size_t read_size = fread(all_weights, 1, weight_size, f);
#         if (read_size != weight_size) {{ 
#             LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
#         }}
#         fclose(f);
#     }}


#     ~new_net() {{
#         free(all_weights);
#     }}

#     {0}

# private:
#     int weight_size = {1};
#     unsigned char* all_weights = (unsigned char*) malloc(weight_size);
# {2}


# {3}

# }};
# }} //namespace hypertea
# """

#     print(a.format(inference_code, weight_size, weight_defs, op_defs()))


