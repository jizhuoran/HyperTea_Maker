import torch.nn as nn
import torch
import numpy as np
import random

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size):
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.require_grad = False
        self.lstm = nn.GRU(embedding_dim, embedding_dim, num_layers = 1, batch_first = True)



    def forward(self, input):

        embedded = self.embedding(input).view(self.batch_size, -1, self.embedding_dim)

        output, hidden = self.lstm(embedded)

        return output, hidden



class FakeRandomGenerator(object):
    """docstring for FakeRandomGenerator"""
    def __init__(self):
        super(FakeRandomGenerator, self).__init__()
        with open('/home/zrji/hypertea_maker/random_number.txt', 'r') as f:
            self.source_vec = [float(n) for n in f.read().split(' ')]
        self.pos = 0


    def generator_random_list(self, item_numbers):
        l = []
        for _ in range(item_numbers):
            l.append(self.source_vec[self.pos])
            self.pos = (self.pos + 1) % len(self.source_vec)
        return l

    def rn(self, item_shape):
        return np.array(self.generator_random_list(np.prod(item_shape))).reshape(item_shape)


# parameters_lists = []
# op_name = 'lstm1'
# declare = []
# precision = 'float'

# type_name = 'hypertea::RNN_CELL_TYPE::LSTM_CELL'
# weight_num = 4

# def RNN_hooker(module, input_data, output_data):

#     def process_parameter(param_prefix):
#         for direction in (['', '_reverse'] if module.bidirectional else ['']):
#             param_name = param_prefix + direction
#             parameters_lists.append(('{}_{}'.format(op_name, param_name), getattr(module, param_name)))
#             param_in_signature.append('{}_{}'.format(op_name, param_name))


#     batch_size = input_data[0].shape[0]
#     assert batch_size == 1, 'Now RNNs do not support batch size > 1'
    
#     direction_type = 'Bidirectional' if module.bidirectional else 'Unidirectional'

#     cpu_layer_declare = []

#     for i in range(module.num_layers):
        
#         param_in_signature = []

#         weight_size, hidden_size = getattr(module, 'weight_ih_l{}'.format(i)).shape

#         input_size = weight_size // weight_num

#         process_parameter('weight_ih_l{}'.format(i))
#         process_parameter('weight_hh_l{}'.format(i))
#         if module.bias:
#             process_parameter('bias_ih_l{}'.format(i))
#             process_parameter('bias_hh_l{}'.format(i))


#         cpu_layer_declare.append('new hypertea::{}RNN_CPU<{}> ( {}, {}, {}, {} )'.format(direction_type, precision, input_size, hidden_size, ', '.join(param_in_signature), type_name))


    
    
#     cpu_signature  = '''
#         std::vector<hypertea::RNNOp_CPU<{}>* > {{
#             {}
#         }}
#         '''.format(precision, (',\n' + ' '*12).join(cpu_layer_declare))


#     gpu_signature = cpu_signature
#     declare.append({'type':'TanHOp', 'op_name':'lstm', 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})

    

# if __name__ == '__main__':
    
#     lstm = nn.LSTM(64, 32, 3, batch_first = True, bidirectional = True)
    
#     # print(type(lstm))

#     lstm.register_forward_hook(RNN_hooker)
    

#     temp = lstm(torch.ones((1, 5, 64), dtype = torch.float))

#     a = nn.Module()

#     # print(set(dir(lstm)) - set(dir(a)))


# def no_use():
#     rg = FakeRandomGenerator()


#     lstm = nn.LSTM(64, 32, 3, batch_first = True, bidirectional = True)




#     print(dir(lstm))
#     print(lstm.weight_ih_l0.shape)
#     print(lstm.weight_hh_l0.shape)
#     print(lstm.bias_ih_l0.shape)
#     print(lstm.bias_hh_l0.shape)



#     lstm.weight_ih_l0.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     lstm.weight_hh_l0.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     lstm.bias_ih_l0.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.bias_hh_l0.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.weight_ih_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     lstm.weight_hh_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     lstm.bias_ih_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.bias_hh_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)

#     lstm.weight_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     lstm.weight_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     lstm.bias_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.bias_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.weight_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     lstm.weight_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     lstm.bias_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.bias_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)



#     lstm.weight_ih_l2.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     lstm.weight_hh_l2.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     lstm.bias_ih_l2.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.bias_hh_l2.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.weight_ih_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     lstm.weight_hh_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     lstm.bias_ih_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     lstm.bias_hh_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
   



#     # gru.weight_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     # gru.weight_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     # gru.bias_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.bias_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.weight_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     # gru.weight_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     # gru.bias_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.bias_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)


#     # gru.weight_ih_l2.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     # gru.weight_hh_l2.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     # gru.bias_ih_l2.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.bias_hh_l2.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.weight_ih_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     # gru.weight_hh_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     # gru.bias_ih_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.bias_hh_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)


#     # gru.weight_ih_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     # gru.weight_hh_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     # gru.bias_ih_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.bias_hh_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)


#     # gru.weight_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     # gru.weight_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     # gru.bias_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.bias_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)

#     # gru.weight_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
#     # gru.weight_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
#     # gru.bias_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
#     # gru.bias_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)


    


#     # print(np.concatenate((h1, rh1)).shape)

#     # exit(0)

#     input_tensor = torch.tensor(rg.rn((1, 5, 64)), dtype = torch.float)
#     h1 = rg.rn((1, 1, 32))
#     c1 = rg.rn((1, 1, 32))
#     rh1 = rg.rn((1, 1, 32))
#     rc1 = rg.rn((1, 1, 32))
    
#     h2 = rg.rn((1, 1, 32))
#     c2 = rg.rn((1, 1, 32))
#     rh2 = rg.rn((1, 1, 32))
#     rc2 = rg.rn((1, 1, 32))
    
#     h3 = rg.rn((1, 1, 32))
#     c3 = rg.rn((1, 1, 32))
#     rh3 = rg.rn((1, 1, 32))
#     rc3 = rg.rn((1, 1, 32))
    
#     # rh1 = rg.rn((1, 1, 32))
#     # rc1 = rg.rn((1, 1, 32))
#     hidden_tensor = (torch.tensor(np.concatenate((h1, rh1, h2, rh2, h3, rh3)), dtype = torch.float), 
#                      torch.tensor(np.concatenate((c1, rc1, c2, rc2, c3, rc3)), dtype = torch.float))



#     temp = lstm(input_tensor, hidden_tensor)

#     # temp = np.concatenate(list(map(lambda x: x.detach().numpy(), temp)))


    

#     # print(temp[1].shape)

#     print(temp[0].reshape(-1))
#     print('---------------------')
#     print(temp[0])
#     print(temp[0].shape)

#     # print(temp[1].reshape(-1))
#     # print(temp.shape)





