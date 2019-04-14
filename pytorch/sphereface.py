# -*- coding: utf-8 -*-
import torch
from sphere20a import sphere20a as Model
from hypertea_generator.hypertea_generator import HyperteaGenerator



model = Model().train()

precision = 'float'

genetator = HyperteaGenerator(model, torch.ones((1, 3, 112, 96), dtype = torch.float), precision)

output = genetator.get_net_output()

inference_code = f'''
    
    void inference( std::vector<{precision}> &data_from_user, std::vector<{precision}> &data_to_user) {{
        
        auto x = DeviceTensor(data_from_user);


        x = relu1_1(conv1_1(x))
        x = x + relu1_3(conv1_3(relu1_2(conv1_2(x))))

        x = relu2_1(conv2_1(x))
        x = x + relu2_3(conv2_3(relu2_2(conv2_2(x))))
        x = x + relu2_5(conv2_5(relu2_4(conv2_4(x))))

        x = relu3_1(conv3_1(x))
        x = x + relu3_3(conv3_3(relu3_2(conv3_2(x))))
        x = x + relu3_5(conv3_5(relu3_4(conv3_4(x))))
        x = x + relu3_7(conv3_7(relu3_6(conv3_6(x))))
        x = x + relu3_9(conv3_9(relu3_8(conv3_8(x))))

        x = relu4_1(conv4_1(x))
        x = x + relu4_3(conv4_3(relu4_2(conv4_2(x))))

        x = fc5(x)
        x = fc6(x)

        x.copy_to_ptr((void*)data_to_user.data());
    }}
'''

print(genetator.network_defination(inference_code, 'work_space/new_net'))







