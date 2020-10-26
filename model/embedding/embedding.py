import torch.nn as nn
import torch

from .type_embedding import PositionalEmbedding, TemporalEmbedding, ModalityEmbedding, DBEmbedding


class PreTrainBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.total_len = args.utter_len + args.sql_len + args.db_len
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

        for item in data:
            assert len(item.split()) == self.total_len

        encoded = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=data, return_tensors='pt',
                                                   add_special_tokens=False)

        output = self.bert(**encoded)  # self.batch_size, self.total_len, -1)
        return output[0]


class InputEmbedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.total_len = args.utter_len + args.sql_len + args.db_len
        self.max_turn = args.max_turn
        self.max_table = args.max_table
        self.position_embedding = PositionalEmbedding(d_model=self.input_size, total_len=self.total_len)

        self.temporal_embedding = TemporalEmbedding(max_turn=self.turn_num, embed_size=self.input_size)

        self.modality_embedding = ModalityEmbedding(embed_size=self.input_size)

        self.db_embedding = DBEmbedding(max_table=self.max_table, embed_size=self.input_size)
        if args.bert:
            self.pre_train_embedding = PreTrainBert(args)

    def parse_batch_content(self, batch_content):
        '''
        :param batch_content: [str1,str2]
        :return: bs,total,hidden
        '''
        batch_size = len(batch_content)
        batch_content_embedding = self.pre_train_embedding(batch_content)
        assert batch_content_embedding.shape == (batch_size, self.total_len, self.input_size)
        return batch_content_embedding

    def parse_content(self, content):
        '''
        :param content: [str1]
        :return: total_len,hidden
        '''
        assert len(content) == 1
        content_embedding = self.pre_train_embedding([content])[0, :, :]
        assert content_embedding.shape == (self.total_len, self.input_size)
        return content_embedding

    def parse_signal(self, signal, type):
        if type == 'temporal_signal':
            return self.temporal_embedding(signal)
        elif type == 'modality_signal':
            return self.modality_embedding(signal)
        elif type == 'position_signal':
            return self.position_embedding(signal)
        elif type == 'db_signal':
            return self.db_embedding(signal)
        else:
            raise Exception('Invalid Signal Type')

    def forward(self, data):
        print('You had better not use this func')
        content_embedding = self.pre_train_embedding(data['content'])
        position_embedding = self.position_embedding(content_embedding)
        temporal_embedding = self.position_embedding(data['temporal_signal'])
        modality_embedding = self.position_embedding(data['modality_signal'])
        db_embedding = self.position_embedding(data['db_signal'])
        signal_embedding = position_embedding + temporal_embedding + modality_embedding + db_embedding

        return torch.cat((content_embedding, signal_embedding), dim=-1)


class OutputEmbedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dict = ['[PAD]', '[SEP]', '=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group by', 'order by',
                     'distinct', 'and', 'limit value', 'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<',
                     'sum', 'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!=', 'union', 'between', '-',
                     '+', '/']


        self.embedding = nn.Embedding(num_embeddings=len(self.dict), embedding_dim=args.hidden, padding_idx=0)
        self.transform_keyword_dist = nn.Linear(self.hidden, len(self.dict))

    def convert_str_to_embedding(self, item):
        '''

        :param item: a str token
        :return:
        '''
        if item in self.dict:
            return self.embedding(self.dict.index(item)).squeeze()
        else:
            raise Exception('This token {} is not in output embedding dict!'.format(item))

    def find_str_idx(self, item):
        '''

        :param item: a str token
        :return:
        '''
        if item in self.dict:
            return self.dict.index(item)
        else:
            raise Exception('This token {} is not in output embedding dict!'.format(item))

    def convert_embedding_to_dist(self, features):
        '''

        :param features: (bs,decoder_len,hidden)
        :return: (bs,decoder_len,len(keywords))
        '''
        return self.transform_keyword_dist(features)


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
