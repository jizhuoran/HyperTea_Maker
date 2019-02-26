# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

# from collections import namedtuple
# from distutils.version import LooseVersion
# from graphviz import Digraph
from array import array
import numpy as np

from functools import partial, reduce


torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

EMBEDDING_DIM = 512
HIDDEN_DIM = 512
BATCH_SIZE = 64


def prod(l):
    return reduce(lambda x, y: x*y, l)

def list2vecstr(l):
    return 'std::vector<int> {'  + ','.join(map(str, l)) + '}'

def bool2str(x):
    return 'true' if x else 'false'

def bool2inplace_str(x):
    return 'IN_PLACE' if x else 'NOT_IN_PLACE'


class LSTMClassifier(nn.Module):

    def __init__(self):
        super(LSTMClassifier, self).__init__()

        self.conv1      = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding = 4, bias=True)
        self.elu1       = nn.ELU()
        self.bn1        = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.conv2      = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding = 1, bias=True)
        self.elu2       = nn.ELU()
        self.bn2        = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.conv3      = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding = 1, bias=True)
        self.elu3       = nn.ELU()
        self.bn3        = nn.BatchNorm2d(128, eps=1e-05, momentum=0, affine=True, track_running_stats=False)

        self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res1_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res1_relu1 = nn.ReLU()
        self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res1_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res2_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res2_relu1 = nn.ReLU()
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res2_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.res3_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res3_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res3_relu1 = nn.ReLU()
        self.res3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res3_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.res4_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res4_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res4_relu1 = nn.ReLU()
        self.res4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res4_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.res5_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res5_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res5_relu1 = nn.ReLU()
        self.res5_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res5_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.deconv1    = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, groups=1, bias=True)
        self.de_elu1    = nn.ELU()
        self.de_bn1     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.deconv2    = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, groups=1, bias=True)
        self.de_elu2    = nn.ELU()
        self.de_bn2     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.deconv3    = nn.ConvTranspose2d(32, 3, 9, stride=1, padding=4, output_padding=0, groups=1, bias=True)
        self.de_tanh3   = nn.Tanh()

        # self.scale_weight = torch.tensor(scale_weight, dtype = torch.float)
        # self.scale_bias = torch.tensor(scale_bias, dtype = torch.float)

    def forward(self, data):

        temp = self.bn1(self.elu1(self.conv1(data)))
        temp = self.bn2(self.elu2(self.conv2(temp)))
        temp = self.bn3(self.elu3(self.conv3(temp)))


        temp += self.res1_bn2(self.res1_conv2(self.res1_bn1(self.res1_relu1(self.res1_bn1(self.res1_conv1(temp))))))
        temp += self.res2_bn2(self.res2_conv2(self.res2_bn1(self.res2_relu1(self.res2_bn1(self.res2_conv1(temp))))))
        temp += self.res3_bn2(self.res3_conv2(self.res3_bn1(self.res3_relu1(self.res3_bn1(self.res3_conv1(temp))))))
        temp += self.res4_bn2(self.res4_conv2(self.res4_bn1(self.res4_relu1(self.res4_bn1(self.res4_conv1(temp))))))
        temp += self.res5_bn2(self.res5_conv2(self.res5_bn1(self.res5_relu1(self.res5_bn1(self.res5_conv1(temp))))))


        temp = self.de_bn1(self.de_elu1(self.deconv1(temp)))
        temp = self.de_bn2(self.de_elu2(self.deconv2(temp)))
        temp = self.de_tanh3(self.deconv3(temp))

        temp = (temp + 1) * 127.5

        return temp







with open('new_net.weight', 'rb') as f:
    float_array = np.fromstring( f.read(), np.float32 )




model = LSTMClassifier().eval()

model.load_state_dict(torch.load('styte_net'))





a = nn.Module()



parameters = []
declarations = []


def batch_norm_hooker(module, input, output, op_name):


    top_shape = input[0].nelement()
    num, channels = input[0].shape[:2]
    eps = module.eps
    scale_factor = 1
    use_global_stats = module.track_running_stats
    bias_name, bias = op_name+'_bias', module.bias
    weight_name, weight = op_name+'_weight', module.weight

    parameters.append((bias_name, bias))
    parameters.append((weight_name, weight))


    if module.track_running_stats:
        mean_name, var_name = op_name+'_mean', op_name+'_var'
        parameters.append((mean_name, module.running_mean))
        parameters.append((var_name,  module.running_var))
    else:
        mean_name = var_name = 'NULL'


    if module.affine:
        weight_name, bias_name = op_name+'_weight', op_name+'_bias'
        parameters.append((weight_name, module.weight))
        parameters.append((bias_name,  module.bias))
    else:
        weight_name = bias_name = 'NULL'


    cpu_signature = '{{ {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} }}'.format(
                top_shape, bias_name, weight_name, 
                num, channels, eps, scale_factor, 
                use_global_stats, mean_name, var_name,
                weight_name, bias_name)
    gpu_signature = ' '
    declarations.append({'type':'BatchNormOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})



def conv_hooker(module, input_data, output_data, op_name):


    if module.bias is not None:
        bias_name, bias = op_name+'_bias', module.bias
        parameters.append((bias_name, bias))
    else:
        bias_name = 'NULL'


    weight_name, weight = op_name+'_weight', module.weight
    parameters.append((weight_name, weight))

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

    cpu_signature = '{{ {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} }}'.format(
                weight_name, bias_name, groups, bool2str(is_1x1), 
                list2vecstr(kernel_shape), list2vecstr(stride), 
                list2vecstr(padding), list2vecstr(dilation),
                list2vecstr(input_shape), list2vecstr(output_shape),
                bool2str(force_nd_conv))
    gpu_signature = ' '
    declarations.append({'type':conv_type, 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})



def relu_hooker(module, input_data, output_data, op_name):
    cpu_signature = '{{ 0, {} }}'.format(bool2inplace_str(module.inplace))
    gpu_signature = ' '
    declarations.append({'type':'ReLUOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


def elu_hooker(module, input_data, output_data, op_name):
    cpu_signature = '{{ 1, {} }}'.format(bool2inplace_str(module.inplace))
    gpu_signature = ' '
    declarations.append({'type':'ELUOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


def tanh_hooker(module, input_data, output_data, op_name):
    cpu_signature = '{{ NOT_IN_PLACE }}'
    gpu_signature = ' '
    declarations.append({'type':'TanHOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})






hooker_map = {
    torch.nn.modules.conv.Conv2d:conv_hooker,
    torch.nn.modules.conv.ConvTranspose2d:conv_hooker,
    torch.nn.modules.batchnorm.BatchNorm2d:batch_norm_hooker,
    torch.nn.modules.activation.ELU:elu_hooker,
    torch.nn.modules.activation.ReLU:relu_hooker,
    torch.nn.modules.activation.ReLU:relu_hooker
}



# print(type(model.bn1))
# print(type(model.deconv1))
# print(type(model.elu1))
# print(type(model.res1_relu1))
# print(type(model.de_tanh3))

# model.bn1.register_forward_hook(partial(batch_norm_hooker, op_name = 'res1_relu1'))
# model.deconv1.register_forward_hook(partial(conv_hooker, op_name = 'res1_relu1'))
# model.de_elu2.register_forward_hook(partial(elu_hooker, op_name = 'de_elu2'))
# model.de_tanh3.register_forward_hook(partial(tanh_hooker, op_name = 'de_tanh3'))



for name, m in model.named_modules():
    if type(m) in hooker_map:
        m.register_forward_hook(partial(hooker_map[type(m)], op_name = name))




def op_defs():
    for declaration in declarations:

        print('{}_CPU<float> {} = {}_CPU<float> ({});'.format(
            declaration['type'], declaration['op_name'],
            declaration['type'], declaration['cpu_signature'])
        )


def para_defs():

    for parameter in parameters:


        print(parameter[0])
        print(parameter[0], ",,,,,", parameter[1].nelement() * 4)

        print(' ')
        print(' ')
        print(' ')
        print(' ')




# conv1_weight = float_array[0:7776];model.conv1.weight.data = torch.tensor(conv1_weight, dtype = torch.float).reshape(model.conv1.weight.data.shape)
# conv1_bias = float_array[7776:7808];model.conv1.bias.data = torch.tensor(conv1_bias, dtype = torch.float).reshape(model.conv1.bias.data.shape)
# scale1_scale = float_array[7808:7840];model.bn1.weight.data = torch.tensor(scale1_scale, dtype = torch.float).reshape(model.bn1.weight.data.shape)
# scale1_bias = float_array[7840:7872];model.bn1.bias.data = torch.tensor(scale1_bias, dtype = torch.float).reshape(model.bn1.bias.data.shape)
# conv2_weight = float_array[7872:40640];model.conv2.weight.data = torch.tensor(conv2_weight, dtype = torch.float).reshape(model.conv2.weight.data.shape)
# conv2_bias = float_array[40640:40704];model.conv2.bias.data = torch.tensor(conv2_bias, dtype = torch.float).reshape(model.conv2.bias.data.shape)
# scale2_scale = float_array[40704:40768];model.bn2.weight.data = torch.tensor(scale2_scale, dtype = torch.float).reshape(model.bn2.weight.data.shape)
# scale2_bias = float_array[40768:40832];model.bn2.bias.data = torch.tensor(scale2_bias, dtype = torch.float).reshape(model.bn2.bias.data.shape)
# conv3_weight = float_array[40832:171904];model.conv3.weight.data = torch.tensor(conv3_weight, dtype = torch.float).reshape(model.conv3.weight.data.shape)
# conv3_bias = float_array[171904:172032];model.conv3.bias.data = torch.tensor(conv3_bias, dtype = torch.float).reshape(model.conv3.bias.data.shape)
# scale3_scale = float_array[172032:172160];model.bn3.weight.data = torch.tensor(scale3_scale, dtype = torch.float).reshape(model.bn3.weight.data.shape)
# scale3_bias = float_array[172160:172288];model.bn3.bias.data = torch.tensor(scale3_bias, dtype = torch.float).reshape(model.bn3.weight.data.shape)
# res1_conv1_weight = float_array[172288:319744];model.res1_conv1.weight.data = torch.tensor(res1_conv1_weight, dtype = torch.float).reshape(model.res1_conv1.weight.data.shape)
# res1_scale1_scale = float_array[319744:319872];model.res1_bn1.weight.data = torch.tensor(res1_scale1_scale, dtype = torch.float).reshape(model.res1_bn1.weight.data.shape)
# res1_scale1_bias = float_array[319872:320000];model.res1_bn1.bias.data = torch.tensor(res1_scale1_bias, dtype = torch.float).reshape(model.res1_bn1.bias.data.shape)
# res1_conv2_weight = float_array[320000:467456];model.res1_conv2.weight.data = torch.tensor(res1_conv2_weight, dtype = torch.float).reshape(model.res1_conv2.weight.data.shape)
# res1_scale2_scale = float_array[467456:467584];model.res1_bn2.weight.data = torch.tensor(res1_scale2_scale, dtype = torch.float).reshape(model.res1_bn2.weight.data.shape)
# res1_scale2_bias = float_array[467584:467712];model.res1_bn2.bias.data = torch.tensor(res1_scale2_bias, dtype = torch.float).reshape(model.res1_bn2.bias.data.shape)
# res2_conv1_weight = float_array[467712:615168];model.res2_conv1.weight.data = torch.tensor(res2_conv1_weight, dtype = torch.float).reshape(model.res2_conv1.weight.data.shape)
# res2_scale1_scale = float_array[615168:615296];model.res2_bn1.weight.data = torch.tensor(res2_scale1_scale, dtype = torch.float).reshape(model.res2_bn1.weight.data.shape)
# res2_scale1_bias = float_array[615296:615424];model.res2_bn1.bias.data = torch.tensor(res2_scale1_bias, dtype = torch.float).reshape(model.res2_bn1.bias.data.shape)
# res2_conv2_weight = float_array[615424:762880];model.res2_conv2.weight.data = torch.tensor(res2_conv2_weight, dtype = torch.float).reshape(model.res2_conv2.weight.data.shape)
# res2_scale2_scale = float_array[762880:763008];model.res2_bn2.weight.data = torch.tensor(res2_scale2_scale, dtype = torch.float).reshape(model.res2_bn2.weight.data.shape)
# res2_scale2_bias = float_array[763008:763136];model.res2_bn2.bias.data = torch.tensor(res2_scale2_bias, dtype = torch.float).reshape(model.res2_bn2.bias.data.shape)
# res3_conv1_weight = float_array[763136:910592];model.res3_conv1.weight.data = torch.tensor(res3_conv1_weight, dtype = torch.float).reshape(model.res3_conv1.weight.data.shape)
# res3_scale1_scale = float_array[910592:910720];model.res3_bn1.weight.data = torch.tensor(res3_scale1_scale, dtype = torch.float).reshape(model.res3_bn1.weight.data.shape)
# res3_scale1_bias = float_array[910720:910848];model.res3_bn1.bias.data = torch.tensor(res3_scale1_bias, dtype = torch.float).reshape(model.res3_bn1.bias.data.shape)
# res3_conv2_weight = float_array[910848:1058304];model.res3_conv2.weight.data = torch.tensor(res3_conv2_weight, dtype = torch.float).reshape(model.res3_conv2.weight.data.shape)
# res3_scale2_scale = float_array[1058304:1058432];model.res3_bn2.weight.data = torch.tensor(res3_scale2_scale, dtype = torch.float).reshape(model.res3_bn2.weight.data.shape)
# res3_scale2_bias = float_array[1058432:1058560];model.res3_bn2.bias.data = torch.tensor(res3_scale2_bias, dtype = torch.float).reshape(model.res3_bn2.bias.data.shape)
# res4_conv1_weight = float_array[1058560:1206016];model.res4_conv1.weight.data = torch.tensor(res4_conv1_weight, dtype = torch.float).reshape(model.res4_conv1.weight.data.shape)
# res4_scale1_scale = float_array[1206016:1206144];model.res4_bn1.weight.data = torch.tensor(res4_scale1_scale, dtype = torch.float).reshape(model.res4_bn1.weight.data.shape)
# res4_scale1_bias = float_array[1206144:1206272];model.res4_bn1.bias.data = torch.tensor(res4_scale1_bias, dtype = torch.float).reshape(model.res4_bn1.bias.data.shape)
# res4_conv2_weight = float_array[1206272:1353728];model.res4_conv2.weight.data = torch.tensor(res4_conv2_weight, dtype = torch.float).reshape(model.res4_conv2.weight.data.shape)
# res4_scale2_scale = float_array[1353728:1353856];model.res4_bn2.weight.data = torch.tensor(res4_scale2_scale, dtype = torch.float).reshape(model.res4_bn2.weight.data.shape)
# res4_scale2_bias = float_array[1353856:1353984];model.res4_bn2.bias.data = torch.tensor(res4_scale2_bias, dtype = torch.float).reshape(model.res4_bn2.bias.data.shape)
# res5_conv1_weight = float_array[1353984:1501440];model.res5_conv1.weight.data = torch.tensor(res5_conv1_weight, dtype = torch.float).reshape(model.res5_conv1.weight.data.shape)
# res5_scale1_scale = float_array[1501440:1501568];model.res5_bn1.weight.data = torch.tensor(res5_scale1_scale, dtype = torch.float).reshape(model.res5_bn1.weight.data.shape)
# res5_scale1_bias = float_array[1501568:1501696];model.res5_bn1.bias.data = torch.tensor(res5_scale1_bias, dtype = torch.float).reshape(model.res5_bn1.bias.data.shape)
# res5_conv2_weight = float_array[1501696:1649152];model.res5_conv2.weight.data = torch.tensor(res5_conv2_weight, dtype = torch.float).reshape(model.res5_conv2.weight.data.shape)
# res5_scale2_scale = float_array[1649152:1649280];model.res5_bn2.weight.data = torch.tensor(res5_scale2_scale, dtype = torch.float).reshape(model.res5_bn2.weight.data.shape)
# res5_scale2_bias = float_array[1649280:1649408];model.res5_bn2.bias.data = torch.tensor(res5_scale2_bias, dtype = torch.float).reshape(model.res5_bn2.bias.data.shape)
# deconv5_1_weight = float_array[1649408:1780480];model.deconv1.weight.data = torch.tensor(deconv5_1_weight, dtype = torch.float).reshape(model.deconv1.weight.data.shape)
# deconv5_1_bias = float_array[1780480:1780544];model.deconv1.bias.data = torch.tensor(deconv5_1_bias, dtype = torch.float).reshape(model.deconv1.bias.data.shape)
# deconv5_1_bn_sc_scale = float_array[1780544:1780608];model.de_bn1.weight.data = torch.tensor(deconv5_1_bn_sc_scale, dtype = torch.float).reshape(model.de_bn1.weight.data.shape)
# deconv5_1_bn_sc_bias = float_array[1780608:1780672];model.de_bn1.bias.data = torch.tensor(deconv5_1_bn_sc_bias, dtype = torch.float).reshape(model.de_bn1.bias.data.shape)
# deconv5_2_weight = float_array[1780672:1813440];model.deconv2.weight.data = torch.tensor(deconv5_2_weight, dtype = torch.float).reshape(model.deconv2.weight.data.shape)
# deconv5_2_bias = float_array[1813440:1813472];model.deconv2.bias.data = torch.tensor(deconv5_2_bias, dtype = torch.float).reshape(model.deconv2.bias.data.shape)
# deconv5_2_bn_sc_scale = float_array[1813472:1813504];model.de_bn2.weight.data = torch.tensor(deconv5_2_bn_sc_scale, dtype = torch.float).reshape(model.de_bn2.weight.data.shape)
# deconv5_2_bn_sc_bias = float_array[1813504:1813536];model.de_bn2.bias.data = torch.tensor(deconv5_2_bn_sc_bias, dtype = torch.float).reshape(model.de_bn2.bias.data.shape)


# deconv5_3_weight = float_array[1813536:1821312];model.deconv3.weight.data = torch.tensor(deconv5_3_weight, dtype = torch.float).reshape(model.deconv3.weight.data.shape)
# deconv5_3_bias = float_array[1821312:1821315];model.deconv3.bias.data = torch.tensor(deconv5_3_bias, dtype = torch.float).reshape(model.deconv3.bias.data.shape)




import cv2
im = np.array(cv2.imread("HKU.jpg"))

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im = im.transpose(2,0,1)



# im = np.concatenate((im[1,:,:], im[0,:,:], im[2,:,:]), axis=0)

output = model(torch.tensor(im, dtype = torch.float).reshape(1, 3, 512, 512))




image_scale1_scale = float_array[1821315:1821318]
image_scale1_bias = float_array[1821318:1821321]
image_scale2_scale = float_array[1821321:1821324]

print(image_scale1_scale)
print(image_scale1_bias)
print(image_scale2_scale)

# for i in range(3):
#     output[:,i,:,:] = output[:,i,:,:] * image_scale1_scale[i] + image_scale1_bias[i]

# output = (output + 1) * 127.5

print(output.reshape(3, 512, 512).detach().numpy())

img = output.reshape(3, 512, 512).detach().numpy().transpose(1,2,0)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


cv2.imwrite('output.jpg', img)

