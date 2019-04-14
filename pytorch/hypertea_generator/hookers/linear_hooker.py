from hypertea_generator.util import *


class LinearHooker(object):
    """docstring for LinearHooker"""
    def __init__(self):
        super(LinearHooker, self).__init__()
    
    
    def linear_hooker(module, input_data, output_data, op_name, declare, params):

        weight_name = '&{}_weight'.format(op_name)
        params.append((weight_name[1:], module.weight))

        if module.bias is not None:
            bias_name = '&{}_bias'.format(op_name)
            params.append((bias_name[1:], module.bias))
        else:
            bias_name = 'nullptr'


        print(module.weight.data.shape)
        print(module.in_features)
        print(module.out_features)

        print(len(input_data))
        print(input_data[0].shape)
        print(output_data[0].shape)

        signature = ' {}, {}, {}, {} '.format(weight_name, bias_name, module.in_features, module.out_features)
        declare.append({'type':'Linear', 'op_name':op_name, 'signature':signature})

