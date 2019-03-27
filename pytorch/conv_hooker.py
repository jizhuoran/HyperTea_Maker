from util import *
from libdnn import Libdnn, LibdnnGenerator


class ConvHooker(object):
    """docstring for ConvHooker"""
    def __init__(self):
        super(ConvHooker, self).__init__()
    

    def conv_hooker(module, input_data, output_data, op_name, declare, params, opencl_collector):


        if module.bias is not None:
            bias_name, bias = op_name+'_bias', module.bias
            params.append((bias_name, bias))
        else:
            bias_name = 'NULL'


        weight_name, weight = op_name+'_weight', module.weight
        params.append((weight_name, weight))

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

        cpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                    weight_name, bias_name, groups, bool2str_(is_1x1), 
                    list2vecstr_(kernel_shape), list2vecstr_(stride), 
                    list2vecstr_(padding), list2vecstr_(dilation),
                    list2vecstr_(input_shape), list2vecstr_(output_shape),
                    bool2str_(force_nd_conv))


        libdnn = Libdnn(op_name+'_forward', groups,
                                conv_type, module.bias is not None, 
                                input_shape, output_shape, 
                                kernel_shape, padding, stride, dilation)

        opencl_collector += libdnn.generate_libdnn_code()


        gpu_signature = '"{}_forward", {}, {}, {}, {}, {}'.format(
            op_name, prod_(output_shape), 
            weight_name, bias_name,
            list2vecstr_(libdnn.local_shape()),
            list2vecstr_(libdnn.global_shape()))

        declare.append({'type':conv_type, 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})



    

    def native_conv_hooker(module, input_data, output_data, op_name, declare, params, opencl_collector):

        assert module.groups == 1, "Native deconv only support groups == 1"
        assert module.transposed, "We only test deconvolution at THIS stage"

        conv_type = "NativeDeconvolutionOp"

        if module.bias is not None:
            bias_name, bias = op_name+'_bias', module.bias
            params.append((bias_name, bias))
        else:
            bias_name = 'NULL'


        weight_name, weight = op_name+'_weight', module.weight
        params.append((weight_name, weight))

        kernel_shape = list(module.kernel_size)
        stride = list(module.stride)
        padding = list(module.padding)
        dilation = list(module.dilation)

        out_channels = output_data[0].shape[0]


        is_1x1 = all(list(map(lambda x: x==1, kernel_shape + stride)) + list(map(lambda x: x==0, padding)))

        if not is_1x1:
            col_h = input_data[0].shape[-2]
            col_w = input_data[0].shape[-1]
            wei_h = weight.shape[-2]
            wei_w = weight.shape[-1]
            pad_h = padding[0]
            pad_w = padding[1]
            stride_h = stride[0]
            stride_w = stride[1]
            dilation_h = dilation[0]
            dilation_w = dilation[1]
            height = output_data[0].shape[-2]
            width = output_data[0].shape[-1]

            col2im_kernel = f'''
__kernel void {op_name}_col2Im(global Dtype* col,
                     global Dtype* im,
                     const int im_offset) {{
    
    global Dtype* im_off = im + im_offset;

    int gid               = (int)get_global_id(0);

    int im_ch  = gid / ({width} * {height});
    int im_pix = gid % ({width} * {height});
    int im_h   = (im_pix / {width}) + {pad_h};
    int im_w   = (im_pix % {width}) + {pad_w};

    int start_h = (im_h < {dilation_h} * ({wei_h} - 1) + 1)
                      ? 0
                      : (im_h - ({dilation_h} * ({wei_h} - 1) + 1)) / {stride_h} + 1;
    int end_h   = min({col_h}, im_h / {stride_h} + 1);
    int start_w = (im_w < {dilation_w} * ({wei_w} - 1) + 1)
                      ? 0
                      : (im_w - ({dilation_w} * ({wei_w} - 1) + 1)) / {stride_w} + 1;
    int end_w = min({col_w}, im_w / {stride_w} + 1);

    int ch_offset = im_ch * {col_w} * {col_h} * {wei_w} * {wei_h};
    col += ch_offset;

    Dtype tmp = (Dtype)0;
    for(int cy = start_h; cy < end_h; cy++)
    {{
        for(int cx = start_w; cx < end_w; cx++)
        {{
            if((im_h - cy * {stride_h}) % {dilation_h} == 0 && (im_w - cx * {stride_w}) % {dilation_w} == 0)
            {{
                int col_off_y = cy + (((im_h - cy * {stride_h}) / {dilation_h}) * {wei_w} * {col_h});
                int col_off_x = cx + (((im_w - cx * {stride_w}) / {dilation_w}) * {col_w} * {col_h});

                tmp += (Dtype)(col[col_off_y * {col_w} + col_off_x]);
            }}
        }}
    }}
    im_off[gid] = tmp;
}}
        '''
            
            opencl_collector += col2im_kernel



        input_shape = list(input_data[0].shape)
        output_shape = [len(output_data)] + list(output_data[0].shape)



        gpu_signature = '"{}_col2Im", {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
            op_name, prod_(list(output_data[0].shape)), out_channels * prod_(kernel_shape),
            weight_name, bias_name,
            list2vecstr_(input_shape),
            list2vecstr_(output_shape),
            bool2str_(is_1x1),
            list2vecstr_([256, 1, 1]),
            list2vecstr_([prod_(list(output_data[0].shape)), 1, 1]))

        declare.append({'type':conv_type, 'op_name':op_name, 'gpu_signature':gpu_signature})


        



        