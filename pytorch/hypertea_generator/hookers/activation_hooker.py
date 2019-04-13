from hypertea_generator.util import *


class ActivationHooker(object):
    """docstring for ActivationHooker"""
    def __init__(self):
        super(ActivationHooker, self).__init__()
    
    
    def relu_hooker(module, input_data, output_data, op_name, declare):
        cpu_signature = ' 0, {} '.format(bool2inplace_str_(module.inplace))
        gpu_signature = cpu_signature
        declare.append({'type':'ReLUOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


    def elu_hooker(module, input_data, output_data, op_name, declare):
        cpu_signature = ' 1, {} '.format(bool2inplace_str_(module.inplace))
        gpu_signature = cpu_signature
        declare.append({'type':'ELUOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


    def tanh_hooker(module, input_data, output_data, op_name, declare):
        cpu_signature = ' NOT_IN_PLACE '
        gpu_signature = cpu_signature
        declare.append({'type':'TanHOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})

    