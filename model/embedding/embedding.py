# TODO: 导入预训练bert 输入并得到input的表示
# TODO：创建不同类型的位置embedding，并加和
import torch.nn as nn
import torch
import math

from .type_embedding import PositionalEmbedding, TemporalEmbedding, ModalityEmbedding, DBEmbedding


class PreTrainBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.total_len = args.total_len
        from transformers import BertModel, BertTokenizer, BertConfig
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, data):
        '''
        get bert pre model
        :param data: ['[CLS] i love you [SEP]', '[CLS] you love me [SEP]']  batch of sentence; the
        sentence should add special token before
        :return: shape: bs, total_len, hidden state of bert
        '''
        assert len(data) == self.batch_size
        for item in data:
            assert len(item.split()) == self.total_len

        encoded = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=data, return_tensors='pt',
                                                   add_special_tokens=False)

        output = self.bert(**encoded)  # self.batch_size, self.total_len, -1)
        return output[0]


class InputEmbedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.hidden = args.hidden
        self.total_len = args.total_len
        self.max_turn = args.turn
        self.position_embedding = PositionalEmbedding(d_model=self.hidden, total_len=self.total_len)

        self.temporal_embedding = TemporalEmbedding(max_turn=self.max_turn, embed_size=self.hidden)

        self.modality_embedding = ModalityEmbedding(embed_size=self.hidden)

        self.db_embedding = DBEmbedding(embed_size=self.hidden)
        if args.bert:
            self.pre_train_embedding = PreTrainBert(args)

    def forward(self, data):
        content_embedding = self.pre_train_embedding(data['content'])

        position_embedding = self.position_embedding(content_embedding)
        temporal_embedding = self.position_embedding(data['temporal_signal'])
        modality_embedding = self.position_embedding(data['modality_signal'])
        db_embedding = self.position_embedding(data['db_signal'])
        signal_embedding = position_embedding + temporal_embedding + modality_embedding + db_embedding
        # TODO:这里采用了bert表示和符号表示拼接的方式，以后可以再考虑
        return torch.cat((content_embedding, signal_embedding), dim=-1)


class OutputEmbedding(nn.Module):
    # TODO：解决计算输出概率的问题
    def __init__(self, args):
        super().__init__()

    def forward(self, ):
        pass


if __name__ == '__main__':
    from collections import namedtuple

    Args = namedtuple('Args', 'batch_size total_len')
    args = Args(2, 5)
    print(args.batch_size, args.total_len)
    bert = PreTrainBert(args)
    data = ['[CLS] i love you [SEP]', '[CLS] you love me [SEP]']
    print(bert(data))
    '''
    {'input_ids': [[101, 1045, 2293, 2017, 102], [101, 2017, 2293, 2033, 102]],
    '''
