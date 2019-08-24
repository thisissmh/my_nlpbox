import torch
from torch import nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence as pack 
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class DynamicLSTM(nn.Module):
    """given a mask matrix and compute the dynamic rnn with fixed seq_length
    """
    def __init__(self, input_dim, hidden_dim, device='cpu', bidirectional=False):
        super(DynamicLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.device = device
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            bidirectional=bidirectional, batch_first=True)
    
    def forward(self, x, mask):
        ori_shape = x.shape
        self.input_type = x.type()
        sorted_x, length, bp = self._sort_by_length(x, mask)
        packed_seq = pack(sorted_x, length, batch_first=True)
        lstm_out, _ = self.lstm(packed_seq)
        unpacked_seq, _ = unpack(lstm_out, batch_first=True)
        output = unpacked_seq[:, bp, :]
        return self._to_original_shape(output, ori_shape)

    
    def _sort_by_length(self, x, mask):
        zero_num = (mask==0).sum(dim=1)
        length = mask.sum(dim=1)
        _, sorted_ind = zero_num.sort()
        sorted_tensor = x[:, sorted_ind, :]
        _, back_pointer = sorted_ind.sort()
        return sorted_tensor, length[sorted_ind], back_pointer
    
    def to_original_shape(self, seq, ori_shape):
        #TODO: fill the changed seq_len of x to original shape
        batch_size, ori_seq_len, dk = ori_shape
        cur_seq_len = seq.shape[1]
        if ori_seq_len == cur_seq_len:
            return seq
        else:
            dif = ori_seq_len - cur_seq_len
            pad_zeros = torch.zeros(batch_size, dif, dk).type(self.input_type)
            output = torch.cat([seq, pad_zeros], dim=1)
            return output
        


class Mask(nn.Module):
    """mask layer, output the batch_size * seq_len mask matrix
    """
    def __init__(self, device):
        super(Mask, self).__init__()
        self.torch = torch if device == 'cpu' else torch.cuda

    def forward(self, inputs):
        output = (inputs!=0).astype(self.torch.FloatTensor)
        return output.unsqueeze(2)


class ScaleDotAttention(nn.Module):

    def __init__(self):
        super(DotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask):
        """
        q,k,v: batch_size * seq_len * feature_dim
        """
        batch_size, seq_len, dk = q.shape

        # batch_size * seq_len * seq_len
        att = torch.matmul(q, k.permute([0, 2, 1])) / float(dk**0.5)
        # add softmax to attention matrix
        att = self.softmax(att)

        # dot product with v
        output = torch.matmul(att, v)
        return output * mask


class MultiHeadAttention(nn.Module):
    """Implementation of multi-head self attention from 'attention is all you need'
    args:
        feature_dim: int, from (batch_size, seq_len, feature_dim)
        head_num: int, num of multi head
    """
    def __init__(self, feature_dim, head_num=4):
        super(MultiHeadAttention, self).__init__()

        self.head_num = head_num
        self.feature_dim = feature_dim
        self.dim_per_head = int(feature_dim / head_num)
        for i in range(head_num):
            setattr(self, 'wq'+str(i), nn.Linear(feature_dim, self.dim_per_head))
            setattr(self, 'wk'+str(i), nn.Linear(feature_dim, self.dim_per_head))
            setattr(self, 'wv'+str(i), nn.Linear(feature_dim, self.dim_per_head))

        self.dot_attention_layer = DotAttention()

    def forward(self, q, k, v, mask):

        sub_q = []
        sub_k = []
        sub_v = []

        #use linear projection, compose the feature_dim to dim_per_head
        for i in range(self.head_num):
            sub_q.append(getattr(self, 'wq'+str(i))(q))
            sub_k.append(getattr(self, 'wk'+str(i))(k))
            sub_v.append(getattr(self, 'wv'+str(i))(v))

        #do self-attention separately
        sub_part = []
        for j in range(self.head_num):
            att = self.dot_attention_layer(sub_q[j], sub_k[j], sub_v[j], mask)
            sub_part.append(att)
            
        #concate heads together
        output = torch.cat(sub_part, dim=2)
        return output





