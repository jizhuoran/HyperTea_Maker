import torch

from functools import partial, reduce
from libdnn import LibdnnGenerator
from MIOBatchNorm import MIOBatchNormGenerator

from activation_hooker import ActivationHooker
from conv_hooker import ConvHooker
from batchnorm_hooker import BatchNormHooker
from rnn_hooker import RNNHooker

class NetInfo(object):
    """docstring for NetInfo"""
    def __init__(self, net, input_tensor, precision):
        super(NetInfo, self).__init__()
        
        self.net = net
        self.parameters_ = []
        self.declarations_ = []

        self.precision = precision

        self.libdnn_conv_kernels = []
        self.MIOpen_batchnorm_kernels = []

        self.hooker_map = {
            torch.nn.modules.conv.Conv2d:partial(
                ConvHooker.conv_hooker, 
                params = self.parameters_, 
                declare = self.declarations_, 
                opencl_collector = self.libdnn_conv_kernels
            ),
            
            torch.nn.modules.conv.ConvTranspose2d:partial(
                ConvHooker.conv_hooker, 
                params = self.parameters_, 
                declare = self.declarations_, 
                opencl_collector = self.libdnn_conv_kernels
            ),

            torch.nn.modules.batchnorm.BatchNorm2d:partial(
                BatchNormHooker.MIOpen_batch_norm_hooker, 
                params = self.parameters_, 
                declare = self.declarations_,
                opencl_collector = self.MIOpen_batchnorm_kernels
                # shift_factor = ''#', 64.0, 32.0'
            ),

            torch.nn.modules.rnn.LSTM:partial(
                RNNHooker.LSTM_hooker, 
                params = self.parameters_, 
                declare = self.declarations_, 
                precision = self.precision
            ),


            torch.nn.modules.rnn.GRU:partial(
                RNNHooker.GRU_hooker, 
                params = self.parameters_, 
                declare = self.declarations_, 
                precision = self.precision
            ),
            
            torch.nn.modules.activation.ELU:partial(
                ActivationHooker.elu_hooker, 
                declare = self.declarations_
            ),

            torch.nn.modules.activation.ReLU:partial(
                ActivationHooker.relu_hooker, 
                declare = self.declarations_
            ),

            torch.nn.modules.activation.Tanh:partial(
                ActivationHooker.tanh_hooker, 
                declare = self.declarations_
            )
        }

        self.attach_hooker_()
        self.net_output = self.forward_to_collect_info_(input_tensor)

        self.libdnn_conv_generator = LibdnnGenerator(self.precision, self.libdnn_conv_kernels)
        self.mio_bn_generator = MIOBatchNormGenerator(self.precision, self.MIOpen_batchnorm_kernels)


    def get_declare_with_param(self):
        return self.declarations_, self.parameters_

    def get_opencl_kernels(self):
        return self.libdnn_conv_generator.generate_conv_code(), self.mio_bn_generator.generate_bn_code()


    def forward_to_collect_info_(self, input_tensor):

        return self.net(input_tensor)
    
    def attach_hooker_(self):
        for name, module in self.net.named_modules():
            if type(module) in self.hooker_map:
                module.register_forward_hook(partial(self.hooker_map[type(module)], op_name = name))


    












