from hypertea_generator.util import *


class ActivationHooker(object):
    """docstring for ActivationHooker"""
    def __init__(self):
        super(ActivationHooker, self).__init__()
    
    
    def prelu_hooker(module, input_data, output_data, op_name, declare, params):
        channels = input_data[0].shape[1]
        spatial_dim = prod_(input_data[0].shape[2:])

        weight_name = '&{}_weight'.format(op_name)
        params.append((weight_name[1:], module.weight))

        signature = ' 0, {} '.format(weight_name, channels, spatial_dim, "NOT_IN_PLACE")
        declare.append({'type':'PReLUOp', 'op_name':op_name, 'signature':signature})

    def relu_hooker(module, input_data, output_data, op_name, declare):
        signature = ' 0, {} '.format(bool2inplace_str_(module.inplace))
        declare.append({'type':'ReLUOp', 'op_name':op_name, 'signature':signature})


    def elu_hooker(module, input_data, output_data, op_name, declare):
        signature = ' 1, {} '.format(bool2inplace_str_(module.inplace))
        declare.append({'type':'ELUOp', 'op_name':op_name, 'signature':signature})


    def tanh_hooker(module, input_data, output_data, op_name, declare):
        signature = ' NOT_IN_PLACE '
        declare.append({'type':'TanHOp', 'op_name':op_name, 'signature':signature})

     