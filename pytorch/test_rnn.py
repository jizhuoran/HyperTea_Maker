# -*- coding: utf-8 -*-
import torch
from torch import nn
from hypertea_generator import HyperteaGenerator




model = nn.LSTM(64, 32, 3, batch_first = True, bidirectional = True)
# model.load_state_dict(torch.load('styte_net'))

precision = 'float'


genetator = HyperteaGenerator(model, torch.ones((1, 5, 64), dtype = torch.float), precision)
# genetator.forward_to_collect_info()


output = genetator.get_net_output()

# print(output.reshape(3, 512, 512).detach().numpy())


cpu_code = f'''
    
    void inference( std::vector<{precision}> &data_from_user, std::vector<{precision}> &data_to_user) {{
        
        TensorCPU<{precision}> data(data_from_user);

        auto temp = model(data);

        hypertea_copy(data_to_user.size(), temp.data(), data_to_user.data());

    }}
'''


gpu_code = f'''
    
    void inference( std::vector<{precision}> &data_from_user, std::vector<{precision}> &data_to_user) {{
        
        TensorGPU<{precision}> data(data_from_user);

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

    }}
'''

print(genetator.hypertea_gpu(gpu_code, 'new_net'))


