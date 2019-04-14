# -*- coding: utf-8 -*-
import torch
from net import LSTMClassifier as Model
from hypertea_generator.hypertea_generator import HyperteaGenerator



import cv2
import numpy as np
im = np.array(cv2.imread("work_space/HKU.jpg"))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = im.transpose(2,0,1)

# im = np.concatenate((im, im), axis=0)

print(im.shape) 


model = Model().train()
model.load_state_dict(torch.load('work_space/styte_net'))

precision = 'float'


genetator = HyperteaGenerator(model, torch.tensor(im.reshape(1,3,512,512), dtype = torch.float), precision)

output = genetator.get_net_output()


img = output.reshape(3, 512, 512).detach().numpy().transpose(1,2,0)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


cv2.imwrite('work_space/output.jpg', img)


inference_code = f'''
    
    void inference( std::vector<{precision}> &data_from_user, std::vector<{precision}> &data_to_user) {{
        
        auto data = DeviceTensor(data_from_user);

        auto temp = bn1(outplace_elu(conv1(data)));
        temp = bn2(outplace_elu(conv2(temp)));
        temp = bn3(outplace_elu(conv3(temp)));


        temp += res1_bn2(res1_conv2(outplace_relu(res1_bn1(res1_conv1(temp)))));
        temp += res2_bn2(res2_conv2(outplace_relu(res2_bn1(res2_conv1(temp)))));
        temp += res3_bn2(res3_conv2(outplace_relu(res3_bn1(res3_conv1(temp)))));
        temp += res4_bn2(res4_conv2(outplace_relu(res4_bn1(res4_conv1(temp)))));
        temp += res5_bn2(res5_conv2(outplace_relu(res5_bn1(res5_conv1(temp)))));
        

        temp = de_bn1(outplace_elu(deconv1(temp)));
        temp = de_bn2(outplace_elu(deconv2(temp)));
        temp = outplace_tanh(deconv3(temp));

        temp = (temp + 1) * 127.5;

        temp.copy_to_ptr((void*)data_to_user.data());
    }}
'''

print(genetator.network_defination(inference_code, 'work_space/new_net'))







