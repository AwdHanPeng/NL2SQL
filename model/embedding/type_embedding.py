import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, total_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(total_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, total_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TemporalEmbedding(nn.Embedding):
    def __init__(self, max_turn, embed_size):
        # 0 无, max_turn+1 db, 第一轮 1, 第二轮 2
        super().__init__(max_turn + 2, embed_size, padding_idx=0)


class ModalityEmbedding(nn.Embedding):
    def __init__(self, embed_size):
        # there Modality: DB, SQL, Text
        # 0 无， 1 table 2 column 3 keyword 4 自然语言
        super().__init__(5, embed_size, padding_idx=0)


class DBEmbedding(nn.Embedding):
    def __init__(self, max_table, embed_size):
        # 0 无  ,max_table+1 * ，1 table1 2 table2 3 table3
        super().__init__(max_table + 2, embed_size, padding_idx=0)
