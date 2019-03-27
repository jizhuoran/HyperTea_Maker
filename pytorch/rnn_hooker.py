from util import *
from functools import partial


class RNNHooker(object):
    """docstring for RNNHooker"""
    def __init__(self):
        super(RNNHooker, self).__init__()
    
    

    def RNN_hooker_(module, input_data, output_data, 
                    op_name, declare, params, 
                    precision, 
                    weight_num, type_name):

        def process_parameter(param_prefix):
            for direction in (['', '_reverse'] if module.bidirectional else ['']):
                param_name = param_prefix + direction
                params.append(('{}_{}'.format(op_name, param_name), getattr(module, param_name)))
                param_in_signature.append('{}_{}'.format(op_name, param_name))


        batch_size = input_data[0].shape[0]
        assert batch_size == 1, 'Now RNNs do not support batch size > 1'
        
        direction_type = 'Bidirectional' if module.bidirectional else 'Unidirectional'

        cpu_layer_declare = []

        for i in range(module.num_layers):
            
            param_in_signature = []

            weight_size, hidden_size = getattr(module, 'weight_ih_l{}'.format(i)).shape

            input_size = weight_size // weight_num

            process_parameter('weight_ih_l{}'.format(i))
            process_parameter('weight_hh_l{}'.format(i))
            if module.bias:
                process_parameter('bias_ih_l{}'.format(i))
                process_parameter('bias_hh_l{}'.format(i))


            cpu_layer_declare.append('new hypertea::{}RNN_CPU<{}> ( {}, {}, {}, {} )'.format(direction_type, precision, input_size, hidden_size, ', '.join(param_in_signature), type_name))

        cpu_signature  = '''
            std::vector<hypertea::RNNOp_CPU<{}>* > {{
                {}
            }}
            '''.format(precision, (',\n' + ' '*12).join(cpu_layer_declare))

        gpu_signature = cpu_signature
        declare.append({'type':'TanHOp', 'op_name':'lstm', 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})


    LSTM_hooker = partial(RNN_hooker_, weight_num = 4, type_name = 'hypertea::RNN_CELL_TYPE::LSTM_CELL')
    GRU_hooker = partial(RNN_hooker_, weight_num = 3, type_name = 'hypertea::RNN_CELL_TYPE::GRU_CELL')