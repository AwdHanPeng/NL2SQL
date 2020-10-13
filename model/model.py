import torch.nn as nn

from .embedding import Embedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Model(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of several embeddings
        self.embedding = Embedding()

        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                      num_layers=self.n_layers)

    def forward(self, input, position, modality, temporal, mask):
        '''
        self.model.forward(data["input"], data["positon label"], data["modality label"], data["temporal label"],
        data["mask"] )

        <<<max length of DB + max turn * (max utter length + max sql length) = total sequence length>>>
        <<<[[DB]; [Utter1;SQL1]; [Utter2;SQL2];...[Utter N;ShiftedSQL N]; [PADDING;PADDING]*(max turn-N)]>>>

        Shape of Input/position/modality/temporal: [Batch_size, total sequence length]
        Shape of Mask: [Batch_size, 1, total sequence length, total sequence length]
        :param input: [[DB]; [Utter1;SQL1]; [Utter2;SQL2];...[Utter N;ShiftedSQL N],[PADDING;PADDING]*(max turn-N)]
        :param position:
        :param modality:
        :param temporal:
        :param mask:
        :return:
        '''

        x = self.embedding(input, position, modality, temporal)
        x = self.transformer_encoder.forward(x, mask)

        return x
