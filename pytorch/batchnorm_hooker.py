from util import *
from MIOBatchNorm import MIOBatchNorm, MIOBatchNormGenerator



class BatchNormHooker(object):
    """docstring for BatchNormHooker"""
    def __init__(self):
        super(BatchNormHooker, self).__init__()
        

    def native_batch_norm_hooker(module, input, output, op_name, declare, params, shift_factor):


        top_shape = input[0].nelement()
        num, channels = input[0].shape[:2]
        eps = module.eps
        scale_factor = 1
        use_global_stats = bool2str_(module.track_running_stats)
        bias_name, bias = op_name+'_bias', module.bias
        weight_name, weight = op_name+'_weight', module.weight


        if module.track_running_stats:
            mean_name, var_name = op_name+'_mean', op_name+'_var'
            params.append((mean_name, module.running_mean))
            params.append((var_name,  module.running_var))
        else:
            mean_name = var_name = 'NULL'


        if module.affine:
            weight_name, bias_name = op_name+'_weight', op_name+'_bias'
            params.append((weight_name, module.weight))
            params.append((bias_name,  module.bias))
        else:
            weight_name = bias_name = 'NULL'


        cpu_signature = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                    top_shape, num, channels, eps, scale_factor, 
                    use_global_stats, mean_name, var_name,
                    weight_name, bias_name)
        

        gpu_signature = cpu_signature + shift_factor


        declare.append({'type':'BatchNormOp', 'op_name':op_name, 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


    def MIOpen_batch_norm_hooker(module, input, output, op_name, declare, params, opencl_collector):


        num, channels = input[0].shape[:2]
        eps = module.eps
        bias_name, bias = op_name+'_bias', module.bias
        weight_name, weight = op_name+'_weight', module.weight


        if module.track_running_stats:
            mean_name, var_name = op_name+'_mean', op_name+'_var'
            params.append((mean_name, module.running_mean))
            params.append((var_name,  module.running_var))
        else:
            mean_name = var_name = 'NULL'


        if module.affine:
            weight_name, bias_name = op_name+'_weight', op_name+'_bias'
            params.append((weight_name, module.weight))
            params.append((bias_name,  module.bias))
        else:
            weight_name = bias_name = 'NULL'
        


        mio_bn = MIOBatchNorm(op_name+'_forward', num,
                                channels, prod_(input[0].shape[2:]), eps)

        opencl_collector += mio_bn.generate_MIO_BN_func()


        gpu_signature = '"{}_forward", {}, {}, {}, {}, {}, {}, {}'.format(
            op_name, 
            mean_name, var_name,
            weight_name, bias_name,
            list2vecstr_(mio_bn.local_shape(), 'size_t'),
            list2vecstr_(mio_bn.global_shape(), 'size_t'),
            channels)


        declare.append({'type':'MIOpenBatchNormOp', 'op_name':op_name, 'gpu_signature':gpu_signature})


    