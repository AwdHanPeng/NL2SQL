import torch.nn as nn

from .embedding import InputEmbedding, OutputEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size if args.input_size else None
        self.batch_size = args.batch_size if args.batch_size else None
        self.total_len = args.total_len if args.total_len else None
        self.hidden = args.hidden if args.hidden else None
        self.n_layers = args.n_layers if args.n_layers else None
        self.attn_heads = args.attn_heads if args.attn_heads else None

        # embedding for BERT, sum of several embeddings
        self.input_embedding, self.output_embedding = InputEmbedding(args), OutputEmbedding(args)

        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                      num_layers=self.n_layers)
        self.tranform_layer = nn.Linear(self.input_size, self.hidden)  # this module convert bert dim to out model dim

    def create_mask(self, signal):
        assert signal.size == (self.batch_size, self.total_len)
        return signal.unsqueeze(1).repeat(1, signal.size(1), 1).unsqueeze(1)

    def forward(self, data):
        '''
        self.model.forward(data["input"], data["positon label"], data["modality label"], data["temporal label"],data["db label"]
        data["mask"] )

        <<<max length of DB + max turn * (max utter length + max sql length) = total sequence length>>>
        <<<[[DB]; [Utter1;SQL1]; [Utter2;SQL2];...[Utter N;ShiftedSQL N]; [PADDING;PADDING]*(max turn-N)]>>>

        Shape of Input/position/modality/temporal: [Batch_size, total sequence length]
        Shape of Mask: [Batch_size, 1, total sequence length, total sequence length]
        :param input: [[DB]; [Utter1;SQL1]; [Utter2;SQL2];...[Utter N;ShiftedSQL N],[PADDING;PADDING]*(max turn-N)]
        :param position: -
        :param modality: 0 无， 1 db 2 utter 3 sql
        :param temporal: 0 无， 第一轮：1 ，，，
        :param db 0 无  1 table 2 column 3 key words
        :return:
        '''

        # 1，输入input和所有特殊的标识信息，得到语料的embedding
        x = self.input_embedding(data)
        # 2,对embedding进行维度转换
        x = self.tranform_layer(x)
        # 3,创建mask
        mask = self.create_mask(data['mask'])
        # 4，使用transformer block解析embedding
        x = self.transformer_encoder(x, mask)
        # 5，取最高层的表示，与输出embedding进行乘积，得到每个step上每个候选单词的概率
        x = self.output_embedding(x)
        return x
        '''
        :param position: -
        :param modality: 0 无， 1 table 2 column 3 keyword 4 自然语言
        :param temporal: 0 db， 第一轮：1 ，，，
        :param db 0 无  1 table1 2 table2 3 table3 先不考虑sql
        
        '''
