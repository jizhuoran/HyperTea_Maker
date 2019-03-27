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



def tanh_hooker(module, input_data, output_data):

    print(module.bidirectional)
    print(module.batch_first)
    print(module.input_size)
    print(module.hidden_size)
    print(module.num_layers)

    print(input_data[0].shape)
    print(output_data[1][0].shape)

    cpu_signature = ' NOT_IN_PLACE '
    gpu_signature = cpu_signature
    # declare.append({'type':'TanHOp', 'op_name':'lstm', 'cpu_signature':cpu_signature, 'gpu_signature':gpu_signature})

    

if __name__ == '__main__':
    
    lstm = nn.LSTM(64, 32, 3, batch_first = True, bidirectional = True)
    
    print(type(lstm))

    lstm.register_forward_hook(tanh_hooker)
    

    temp = lstm(torch.ones((2, 5, 64), dtype = torch.float))

    a = nn.Module()

    print(set(dir(lstm)) - set(dir(a)))


def no_use():
    rg = FakeRandomGenerator()


    lstm = nn.LSTM(64, 32, 3, batch_first = True, bidirectional = True)




    print(dir(lstm))
    print(lstm.weight_ih_l0.shape)
    print(lstm.weight_hh_l0.shape)
    print(lstm.bias_ih_l0.shape)
    print(lstm.bias_hh_l0.shape)



    lstm.weight_ih_l0.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    lstm.weight_hh_l0.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    lstm.bias_ih_l0.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.bias_hh_l0.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.weight_ih_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    lstm.weight_hh_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    lstm.bias_ih_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.bias_hh_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)

    lstm.weight_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    lstm.weight_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    lstm.bias_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.bias_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.weight_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    lstm.weight_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    lstm.bias_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.bias_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)



    lstm.weight_ih_l2.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    lstm.weight_hh_l2.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    lstm.bias_ih_l2.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.bias_hh_l2.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.weight_ih_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    lstm.weight_hh_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    lstm.bias_ih_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    lstm.bias_hh_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
   



    # gru.weight_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    # gru.weight_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    # gru.bias_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.bias_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.weight_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    # gru.weight_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    # gru.bias_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.bias_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)


    # gru.weight_ih_l2.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    # gru.weight_hh_l2.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    # gru.bias_ih_l2.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.bias_hh_l2.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.weight_ih_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    # gru.weight_hh_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    # gru.bias_ih_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.bias_hh_l2_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)


    # gru.weight_ih_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    # gru.weight_hh_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    # gru.bias_ih_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.bias_hh_l0_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)


    # gru.weight_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    # gru.weight_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    # gru.bias_ih_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.bias_hh_l1.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)

    # gru.weight_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 64)), rg.rn((32, 64)), rg.rn((32, 64)))), dtype = torch.float)
    # gru.weight_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32, 32)), rg.rn((32, 32)), rg.rn((32, 32)))), dtype = torch.float)
    # gru.bias_ih_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)
    # gru.bias_hh_l1_reverse.data = torch.tensor(np.concatenate((rg.rn((32)), rg.rn((32)), rg.rn((32)))), dtype = torch.float)


    


    # print(np.concatenate((h1, rh1)).shape)

    # exit(0)

    input_tensor = torch.tensor(rg.rn((1, 5, 64)), dtype = torch.float)
    h1 = rg.rn((1, 1, 32))
    c1 = rg.rn((1, 1, 32))
    rh1 = rg.rn((1, 1, 32))
    rc1 = rg.rn((1, 1, 32))
    
    h2 = rg.rn((1, 1, 32))
    c2 = rg.rn((1, 1, 32))
    rh2 = rg.rn((1, 1, 32))
    rc2 = rg.rn((1, 1, 32))
    
    h3 = rg.rn((1, 1, 32))
    c3 = rg.rn((1, 1, 32))
    rh3 = rg.rn((1, 1, 32))
    rc3 = rg.rn((1, 1, 32))
    
    # rh1 = rg.rn((1, 1, 32))
    # rc1 = rg.rn((1, 1, 32))
    hidden_tensor = (torch.tensor(np.concatenate((h1, rh1, h2, rh2, h3, rh3)), dtype = torch.float), 
                     torch.tensor(np.concatenate((c1, rc1, c2, rc2, c3, rc3)), dtype = torch.float))



    temp = lstm(input_tensor, hidden_tensor)

    # temp = np.concatenate(list(map(lambda x: x.detach().numpy(), temp)))


    

    # print(temp[1].shape)

    print(temp[0].reshape(-1))
    print('---------------------')
    print(temp[0])
    print(temp[0].shape)

    # print(temp[1].reshape(-1))
    # print(temp.shape)





