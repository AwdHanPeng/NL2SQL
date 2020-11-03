import torch
import torch.nn as nn


class PreTrainBert(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import BertModel, BertTokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir='./bert')
        self.bert_vocab = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./bert').get_vocab()
        self.unk_idx = self.bert_vocab.get('[UNK]')
        self.pad_idx = self.bert_vocab.get('[PAD]')

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
        data_tokens = torch.tensor(data_tokens)

        output = self.bert(input_ids=data_tokens, attention_mask=(
                data_tokens != self.pad_idx).int())  # self.batch_size, self.total_len, -1)
        return output[0]


def init_embedding_matrix(word_dict, bert):
    embedding_list = []
    tokens = [item.split() for item in word_dict]
    for token in tokens:
        embedding = bert([token]).squeeze(0)
        if embedding.shape[0] > 1:
            embedding = embedding.sum(dim=0)
        else:
            embedding = embedding.squeeze(0)
        embedding_list.append(embedding)
    embeddings = torch.stack(embedding_list, dim=0)
    print('Save embedding matrix into files')
    import pickle
    with open('./embedding.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    return embeddings.detach()


word_dict = ['[PAD]', '[SEP]', '=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group by',
             'order by', 'distinct', 'and', 'limit value', 'limit', 'descent', '>', 'average', 'having',
             'max', 'in', '<', 'sum', 'intersect', 'not', 'min', 'except', 'or', 'ascent', 'like', '!=',
             'union', 'between', '-', '+', '/']
init_embedding_matrix(word_dict, PreTrainBert())
