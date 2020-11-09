import torch.nn as nn
import torch

from .type_embedding import PositionalEmbedding, TemporalEmbedding, ModalityEmbedding, DBEmbedding


class PreTrainBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import BertModel, BertTokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir='./bert')
        self.bert_vocab = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./bert').get_vocab()
        self.unk_idx = self.bert_vocab.get('[UNK]')
        self.pad_idx = self.bert_vocab.get('[PAD]')
        self.cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        if args.save_bert_vocab:
            with open('./bert_vocab.txt', 'w', encoding='UTF-8') as f:
                for key, value in self.bert_vocab.items():
                    f.writelines('{} {}\n'.format(key, value))

    def forward(self, data):
        '''
        get bert pre model
        :param data: [['[CLS]' 'i' 'love' 'you' '[SEP]'], ['[CLS]' 'i' 'love' 'you' '[SEP]'],]  batch of sentence; the
        sentence should add special token before
        :return: shape: bs, total_len, hidden state of bert
        '''
        data_tokens = []

        for item in data:
            data_tokens.append([self.bert_vocab.get(s, self.unk_idx) for s in item])
        data_tokens = torch.tensor(data_tokens).to(self.device)

        output = self.bert(input_ids=data_tokens,
                           attention_mask=(data_tokens != self.pad_idx).int())  # self.batch_size, self.total_len, -1)
        # output = self.bert(input_ids=data_tokens)
        return output[0]


class InputEmbedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.embedding_size = args.hidden
        self.total_len = args.utter_len + args.sql_len + args.db_len
        self.utter_len = args.utter_len
        self.input_size = args.input_size
        self.turn_num = args.turn_num
        self.max_table = args.max_table

        self.cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.position_embedding = PositionalEmbedding(d_model=self.embedding_size, total_len=self.total_len)

        self.temporal_embedding = TemporalEmbedding(max_turn=self.turn_num, embed_size=self.embedding_size)

        self.modality_embedding = ModalityEmbedding(embed_size=self.embedding_size)

        self.db_embedding = DBEmbedding(max_table=self.max_table, embed_size=self.embedding_size)
        if args.use_bert:
            self.pre_train_embedding = PreTrainBert(args)
        self.tranform_layer = nn.Linear(self.input_size,
                                        self.embedding_size)  # this module convert bert dim to out model dim

    def parse_batch_content(self, batch_content):
        '''
        :param batch_content: [[,,,]],[,,,]]]
        :return: bs,total,hidden
        '''
        batch_content_embedding = self.pre_train_embedding(batch_content)

        return self.tranform_layer(batch_content_embedding)

    def parse_content(self, content):
        '''
        :param content: [str1]
        :return: total_len,hidden
        '''

        content_embedding = self.pre_train_embedding([content])[0, :, :]
        return self.tranform_layer(content_embedding)

    def parse_signal(self, signal, type):
        signal = signal.to(self.device)
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
        self.word_dict = ['[PAD]', '[SEP]', '=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group by',
                          'order by', 'distinct', 'and', 'limit value', 'limit', 'descent', '>', 'average', 'having',
                          'max', 'in', '<', 'sum', 'intersect', 'not', 'min', 'except', 'or', 'ascent', 'like', '!=',
                          'union', 'between', '-', '+', '/']
        self.key_feature_init = args.key_feature_init
        self.key_file_init = args.key_file_init
        self.hidden = args.hidden
        self.input_size = args.input_size
        if not args.key_feature_init:
            if args.key_file_init:
                self.tranform_pre_hidden = nn.Linear(args.input_size, args.hidden)
                import pickle
                with open('./model/embedding/embedding.pkl', 'rb') as f:
                    print('Loading embedding files')
                    embedding_matrix = torch.tensor(pickle.load(f))
                    assert embedding_matrix.shape == (len(self.dict), args.input_size)
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
            else:
                self.embedding = nn.Embedding(num_embeddings=len(self.dict), embedding_dim=args.hidden, padding_idx=0)
        self.transform_keyword_dist = nn.Linear(args.hidden, len(self.dict))
        self.cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

    def convert_str_to_embedding(self, item, embedding_matrix=None):
        '''
        :param embedding_matrix:
        :param item: a str token
        :return:
        '''
        if item in self.dict:
            if self.key_feature_init and embedding_matrix is not None:
                assert embedding_matrix.shape == (len(self.dict), self.hidden)
                return embedding_matrix[self.dict.index(item)].to(self.device).squeeze()
            else:
                if self.key_file_init:
                    return self.tranform_pre_hidden(
                        self.embedding(torch.tensor(self.dict.index(item)).to(self.device)).squeeze())
                return self.embedding(torch.tensor(self.dict.index(item)).to(self.device)).squeeze()
        else:
            raise Exception('This token {} is not in output embedding dict!'.format(item))

    def in_dict(self, item):
        if item in self.dict:
            return True
        else:
            return False

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
    from transformers import BertModel, BertTokenizer

    bert = BertModel.from_pretrained('bert-base-uncased')
    bert_vocab = BertTokenizer.from_pretrained('bert-base-uncased').get_vocab()
    data_tokens = torch.tensor([[0, 16273, 16272, 16270], [0, 15570, 15962, 15825]])
    pad_idx = bert_vocab.get('[PAD]')
    print(pad_idx)
    output = bert(input_ids=data_tokens, attention_mask=(data_tokens != pad_idx).int())
    print(output)
