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
    def __init__(self, max_turn, embed_size=512):
        super().__init__(max_turn + 1, embed_size, padding_idx=0)


class ModalityEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        # there Modality: DB, SQL, Text
        super().__init__(4, embed_size, padding_idx=0)


class DBEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        # three DB element: Table && Column && SQL keywords
        super().__init__(4, embed_size, padding_idx=0)
