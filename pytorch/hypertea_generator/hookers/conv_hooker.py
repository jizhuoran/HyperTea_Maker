from hypertea_generator.util import *
from hypertea_generator.opencl_generator.libdnn import Libdnn, LibdnnGenerator


class ConvHooker(object):
    """docstring for ConvHooker"""
    def __init__(self):
        super(ConvHooker, self).__init__()
    

    def libdnn_conv_hooker(module, input_data, output_data, op_name, declare, params, opencl_collector):


        if module.bias is not None:
            bias_name = '&{}_bias'.format(op_name)
            params.append((bias_name[1:], module.bias))
        else:
            bias_name = 'nullptr'


        weight_name = '&{}_weight'.format(op_name)
        params.append((weight_name[1:], module.weight))


        kernel_shape = list(module.kernel_size)
        stride = list(module.stride)
        padding = list(module.padding)
        dilation = list(module.dilation)


        is_1x1 = all(list(map(lambda x: x==1, kernel_shape + stride)) + list(map(lambda x: x==0, padding)))

        input_shape = list(input_data[0].shape)
        output_shape = [len(output_data)] + list(output_data[0].shape)


        conv_type = 'LibDNNDeconvOp' if module.transposed else 'LibDNNConvOp'

        libdnn = Libdnn(op_name+'_forward', module.groups,
                                conv_type, module.bias is not None, 
                                input_shape, output_shape, 
                                kernel_shape, padding, stride, dilation)

        opencl_collector += libdnn.generate_libdnn_code()


        signature = '"{}_forward", {}, {}, {}, {}, {}'.format(
            op_name, prod_(output_shape), 
            weight_name, bias_name,
            list2vecstr_(libdnn.local_shape(), 'size_t'),
            list2vecstr_(libdnn.global_shape(), 'size_t'))

        declare.append({'type':conv_type, 'op_name':op_name, 'signature':signature})



    def native_conv_hooker(module, input_data, output_data, op_name, declare, params):

        assert module.groups == 1, "Native deconv only support groups == 1"

        conv_type = "DeconvolutionOp" if module.transposed else "ConvolutionOp"

        if module.bias is not None:
            bias_name = '&{}_bias'.format(op_name)
            params.append((bias_name[1:], module.bias))
        else:
            bias_name = 'nullptr'


        weight_name = '&{}_weight'.format(op_name)
        params.append((weight_name[1:], module.weight))


        kernel_shape = list(module.kernel_size)
        stride = list(module.stride)
        padding = list(module.padding)
        dilation = list(module.dilation)
        input_shape = list(input_data[0].shape)
        output_shape = [len(output_data)] + list(output_data[0].shape)

        is_1x1 = all(list(map(lambda x: x==1, kernel_shape + stride)) + list(map(lambda x: x==0, padding)))


        signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
            
            weight_name, bias_name,
            module.groups, 
            bool2str_(is_1x1),
            list2vecstr_(kernel_shape), list2vecstr_(stride), 
            list2vecstr_(padding), list2vecstr_(dilation),
            list2vecstr_(input_shape), list2vecstr_(output_shape)
        )


        declare.append({'type':conv_type, 'op_name':op_name, 'signature':signature})


        



        