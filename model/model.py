import torch.nn as nn

from .embedding import InputEmbedding, OutputEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.total_len = args.utter_len + args.sql_len + args.db_len
        self.utter_len, self.db_len, self.sql_len = args.utter_len, args.db_len, args.sql_len

        self.hidden = args.hidden
        self.ffn_dim = args.ffn_dim
        self.n_layers = args.n_layers
        self.attn_heads = args.attn_heads
        self.max_turn = args.max_turn
        self.decode_length = args.decode_length
        self.utterance_rnn_input_size = args.utterrnn_input
        self.utterance_rnn_output_size = args.utterrnn_output
        self.decoder_rnn_input_size = args.decodernn_input
        self.decoder_rnn_output_size = args.decodernn_output

        self.cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        # embedding for BERT, sum of several embeddings
        self.input_embedding, self.output_keyword_embedding = InputEmbedding(args), OutputEmbedding(args)

        if self.n_layers >= 1:
            self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads,
                                                                     dim_feedforward=self.ffn_dim)
            self.transformer_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.n_layers)
        self.tranform_layer = nn.Linear(self.input_size, self.hidden)  # this module convert bert dim to out model dim

        self.decoder_rnn_cell = nn.GRUCell(self.decoder_rnn_input_size, self.decoder_rnn_output_size)
        self.utterance_rnn_cell = nn.GRUCell(self.utterance_rnn_input_size, self.utterance_rnn_output_size)

        # transform for concat(s,l) and then get attention weight from last utterance
        self.tranform_attention = nn.Linear(self.utterance_rnn_output_size + self.decoder_rnn_output_size, self.hidden)
        # transform for concat(s,l,last_utter_sum) and then get attention weight from every turn
        self.tranform_fuse_lastutter = nn.Linear(
            self.utterance_rnn_output_size + self.decoder_rnn_output_size + self.hidden, self.hidden)
        # fuse three modality attention weight
        self.three_fuse = args.three_fuse
        if self.three_fuse:
            self.tranform_fuse_attention = nn.Linear(self.hidden * 3, self.utterance_rnn_input_size)
        else:
            self.tranform_fuse_attention = nn.Linear(self.hidden * 2, self.utterance_rnn_input_size)
        # fuse input sql embedding and utterrnn state for feed into decoder rnn
        # self.tranform_fuse_input_state = nn.Linear(self.utterance_rnn_output_size + self.hidden,
        #                                            self.decoder_rnn_input_size)

        # fuse the concated db feature into hidden dim
        self.trigger_db_fuse_concat = args.db_fuse_concat
        if self.trigger_db_fuse_concat:
            self.tranform_fuse_db_feature = nn.Linear(self.max_turn * self.hidden, self.hidden)

        # fuse table and column to express A.b
        self.table_column_fuse_rnn = nn.GRU(self.hidden, self.hidden // 2, bidirectional=True)
        # we set gru state size == hidden/2, so no need to convert dim
        # self.tranform_fuse_column_table = nn.Linear(self.hidden * 2, self.hidden)

        # transform decoder output state into hidden size, to caculate prob dist
        # we set decoder_rnn_output_size == hidden, so no need to convert dim
        # self.tranform_decoder_output = nn.Linear(self.decoder_rnn_output_size, self.hidden)

        # To get better init for utterrnn, we chose sum of last utter hiddens and convert into rnn size
        # we set utterance_rnn_output_size == hidden, so no need to convert dim
        # self.tranform_lastutter_initutterrnn = nn.Linear(self.hidden, self.utterance_rnn_output_size)

        self.trigger_decode_in_out_fuse = args.decode_in_out_fuse
        self.trigger_db_embedding_feature_bilinear = args.db_embedding_feature_bilinear

        if self.trigger_decode_in_out_fuse:
            self.tranform_fuse_decode_in_out = nn.Linear(self.decoder_rnn_output_size + self.decoder_rnn_input_size,
                                                         self.hidden)
        if self.trigger_db_embedding_feature_bilinear:
            # (turn_batch, decode_length, hidden) * (turn_batch,db_units_num,hidden) ->(turn_batch, decode_length,db_units_num)
            self.db_embedding_feature_bilinear = nn.Linear(self.hidden, self.hidden)
        self.hard_atten = args.hard_atten
        self.pre_trans = args.pre_trans
        self.utter_fuse = args.utter_fuse
        if self.utter_fuse:
            self.tranform_fuse_utter = nn.Linear(self.hidden * 2 + self.utterance_rnn_output_size,
                                                 self.decoder_rnn_input_size)
        self.base_model = args.base_model
        self.use_signal = args.use_signal
        self.embedding_matrix_random = args.embedding_matrix_random
        self.last_db_feature = args.last_db_feature

    def create_pre_turn_embedding(self, data):
        '''
        we should create a turn -1 embedding, its content just include db, not have sql and utter,so [PAD]
        and its Temporal embedding is [:self.db_len]+ other
        its Modality embedding is [:self.db_len]+other
        its DB embedding is [:self.db_len]+other
        The mask is mask
        :param data:
        :return:
        '''

    def mulit_modal_embedding(self, data):
        '''
        get mulit modal embedding for content and sole utterance
        :param data: [dict{},dict{},dict{}]
        :return:
        batch_embedding, # turns+1,total_len,hidden <-------we add a pre turn embedding
        batch_utter_embedding # turns,utter_len,hidden
        '''

        batch_content = [item['content'] for item in data]
        batch_content_embedding = self.input_embedding.parse_batch_content(batch_content,
                                                                           'content')  # turns,total_len,hidden

        if self.pre_trans: batch_content_embedding = self.tranform_layer(batch_content_embedding)

        batch_temporal_embedding, batch_modality_embedding, batch_db_embedding = map(
            lambda type: self.input_embedding.parse_signal(
                torch.tensor([item[type] for item in data]).to(self.device), type),
            ['temporal_signal', 'modality_signal', 'db_signal'])
        batch_positional_embedding = self.input_embedding.parse_signal(batch_content_embedding, 'position_signal')

        batch_signal_embedding = batch_temporal_embedding + batch_modality_embedding + batch_db_embedding + batch_positional_embedding
        if self.use_signal:
            batch_embedding = batch_content_embedding + batch_signal_embedding
        else:
            batch_embedding = batch_content_embedding

        # we get a pre turn embedding, and its content just include db, not have sql and utter,so [PAD]
        # this embedding is used to generate the first turn sql
        pre_turn_content = data[0]['content'][:self.db_len] + ['[PAD]'] * (self.utter_len + self.sql_len)
        pre_turn_content_embedding = self.input_embedding.parse_batch_content([pre_turn_content], 'content').squeeze(0)
        if self.pre_trans: pre_turn_content_embedding = self.tranform_layer(pre_turn_content_embedding)
        pre_turn_signal_embedding = batch_signal_embedding[0]  # just get this, because we have a mask
        pre_turn_embedding = pre_turn_content_embedding + pre_turn_signal_embedding
        pre_turn_embedding = pre_turn_embedding.unsqueeze(0)

        batch_embedding = torch.cat((pre_turn_embedding, batch_embedding), dim=0)  # (turn+1,len,hidden)

        batch_utterance = [item['utter'] for item in data]
        batch_utterance_embedding = self.input_embedding.parse_batch_content(batch_utterance,
                                                                             'utterance')  # turns,utter_len,hidden
        if self.pre_trans: batch_utterance_embedding = self.tranform_layer(batch_utterance_embedding)
        assert batch_utterance_embedding.shape[1] == self.utter_len
        batch_utter_temporal_embedding, batch_utter_modality_embedding, batch_utter_db_embedding = map(
            lambda signal, type: self.input_embedding.parse_signal(torch.tensor(signal).to(self.device), type),
            [[[i + 1] * self.utter_len for i in range(len(data))], [[4] * self.utter_len for i in range(len(data))],
             [[0] * self.utter_len for i in range(len(data))]], ['temporal_signal', 'modality_signal', 'db_signal'])

        batch_utter_positional_embedding = self.input_embedding.parse_signal(batch_utterance_embedding,
                                                                             'position_signal')
        if self.use_signal:
            batch_utter_embedding = batch_utterance_embedding + batch_utter_temporal_embedding + batch_utter_modality_embedding + batch_utter_db_embedding + batch_utter_positional_embedding
        else:
            batch_utter_embedding = batch_utterance_embedding
        return batch_embedding, batch_utter_embedding

    def create_turn_batch(self, turn_feature, turn_mask):
        '''
        convert original turn data into max-turn turn data, and the turn-1 could be viewed as batch_size
        :param turn_feature: (turn+1,len,hidden)
        :param turn_mask: (turn+1,len)
        :return:
        turn_batch_feature: (turn,max_turn,len,hidden);
        turn_batch_mask: (turn,max_turn) -> use to mask padding content for utterlevel rnn
        turn_batch_content_mask: (turn,max_turn,len) -> use to mask content in order to multipath
        '''
        turn_num = turn_feature.shape[0]  # turn+1
        turn_batch_feature = []
        turn_batch_content_mask = []
        turn_batch_mask = []
        for i in range(turn_num - 1):  # we dont need the last content pair
            left, right = (i - self.max_turn + 1) if (i - self.max_turn + 1) > 0 else 0, i
            valid_feature = turn_feature[left:right + 1, :, :]
            valid_content_mask = turn_mask[left:right + 1, :]
            if valid_feature.shape[0] < self.max_turn:
                zero_feature = torch.zeros(
                    [self.max_turn - valid_feature.shape[0], valid_feature.shape[1], valid_feature.shape[2]]).type_as(
                    valid_feature).to(self.device)
                zero_mask = torch.zeros(
                    [self.max_turn - valid_content_mask.shape[0], valid_content_mask.shape[1]]).type_as(
                    valid_content_mask).to(self.device)
                turn_batch_feature.append(torch.cat((zero_feature, valid_feature), dim=0))
                turn_batch_content_mask.append(torch.cat((zero_mask, valid_content_mask), dim=0))
                turn_batch_mask.append(torch.tensor([0] * len(zero_feature) + [1] * len(valid_feature)))

            else:
                turn_batch_feature.append(valid_feature)
                turn_batch_content_mask.append(valid_content_mask)
                turn_batch_mask.append(torch.tensor([1] * len(valid_feature)))
        for item in turn_batch_feature:
            assert item.shape == (self.max_turn, self.total_len, self.hidden)
        for item in turn_batch_content_mask:
            assert item.shape == (self.max_turn, self.total_len)
        for item in turn_batch_mask:
            assert len(item) == self.max_turn
        assert len(turn_batch_mask) == len(turn_batch_feature) == len(turn_batch_content_mask)
        turn_batch_feature = torch.stack(turn_batch_feature, dim=0)
        assert turn_batch_feature.shape == (turn_num - 1, self.max_turn, self.total_len, self.hidden)
        turn_batch_mask = torch.stack(turn_batch_mask, dim=0).to(self.device)
        assert turn_batch_mask.shape == (turn_num - 1, self.max_turn)
        turn_batch_content_mask = torch.stack(turn_batch_content_mask, dim=0)
        assert turn_batch_content_mask.shape == (turn_num - 1, self.max_turn, self.total_len)

        return turn_batch_feature, turn_batch_mask, turn_batch_content_mask

    def hierarchial_decode(self, turn_batch_feature, turn_utter_encoder_feature, decoder_input_sql_embedding,
                           turn_batch_mask, turn_batch_content_mask, turn_utter_mask):
        """
        core two level decode
        :param decoder_input_sql_embedding: (turn_num, self.decode_length, self.hidden)
        :param turn_batch_feature: (turn_num, self.max_turn, self.total_len, self.hidden)
        :param turn_utter_encoder_feature: (turn_num,self.utter_len,self.hidden)
        :param turn_batch_mask: (turn_num,self.max_turn)
        :param turn_batch_content_mask: (turn,max_turn,len) # you also need reverse
        :return: turn_num, self.decode_length, self.decoder_rnn_output_size
        """

        assert decoder_input_sql_embedding.shape[1] == self.decode_length
        decoder_state_list = []
        turn_batch_num = turn_batch_feature.shape[0]

        # we get last utter sum to init the first hidden state of decoder rnn
        # current_decoder_state = torch.zeros(turn_batch_num, self.decoder_rnn_output_size)
        current_decoder_state = torch.sum(
            turn_utter_encoder_feature.masked_fill(turn_utter_mask.unsqueeze(-1) == 0, 0.0), dim=1)

        rever_turn_batch_feature = torch.flip(turn_batch_feature, dims=[1])  # reverse for utterance rnn
        rever_turn_batch_content_mask = torch.flip(turn_batch_content_mask, dims=[1])  # reverse for content mask
        for i in range(self.decode_length):
            # current_utterrnn_state = torch.zeros(turn_batch_num, self.utterance_rnn_output_size)
            # we get last utter sum to init the first hidden state of utterlevel rnn
            # current_utterrnn_state = self.tranform_lastutter_initutterrnn(torch.sum(turn_utter_encoder_feature, dim=1))
            # we set utterance_rnn_output_size == hidden, so no need to convert dim
            current_utterrnn_state = torch.sum(
                turn_utter_encoder_feature.masked_fill(turn_utter_mask.unsqueeze(-1) == 0, 0.0), dim=1)

            utterrnn_state_list = []
            for j in range(self.max_turn):
                attention_key = self.tranform_attention(
                    torch.cat((current_utterrnn_state, current_decoder_state), dim=-1))
                # get s l and atten from last utter
                last_utter_sum = self.attention_sum(attention_key, turn_utter_encoder_feature,
                                                    turn_utter_mask)

                # fuse utter sum and s and l for atten from every turn
                # bs, hidden
                fuse_attention_key = self.tranform_fuse_lastutter(
                    torch.cat((current_utterrnn_state, current_decoder_state, last_utter_sum), dim=-1))

                # split db,utter,sql feature and mask in every turn content
                # bs, db/utter/sql_len, hidden ;; bs, db/utter/sql_len
                db_feature, utter_feature, sql_feature = self.split_feature(rever_turn_batch_feature[:, j, :, :])
                db_mask, utter_mask, sql_mask = self.split_feature(rever_turn_batch_content_mask[:, j, :])

                # get weight sum db,utter,sql feature respectively
                db_atten_sum, utter_atten_sum, sql_atten_sum = map(self.attention_sum, [fuse_attention_key] * 3,
                                                                   [db_feature, utter_feature, sql_feature],
                                                                   [db_mask, utter_mask, sql_mask])
                # fuse the weight sum of db,utter,sql
                if self.three_fuse:
                    fuse_atten_sum = self.tranform_fuse_attention(
                        torch.cat((db_atten_sum, utter_atten_sum, sql_atten_sum),
                                  dim=-1))  # turn_num-1,utterance_rnn_input_size
                # feed mulitpath attn sum into utter rnn
                else:
                    fuse_atten_sum = self.tranform_fuse_attention(torch.cat((utter_atten_sum, sql_atten_sum), dim=-1))
                new_utterrnn_state = self.utterance_rnn_cell(fuse_atten_sum,
                                                             current_utterrnn_state)  # turn_num-1,utterance_rnn_output_size
                # store current utterrnn state
                utterrnn_state_list.append(new_utterrnn_state)
                # update current utterrnn state
                current_utterrnn_state = new_utterrnn_state
            # stack the list of utterrnn state
            utter_state_list = torch.stack(utterrnn_state_list, dim=1)
            utter_state_list = torch.flip(utter_state_list, dims=[1])  # reverse
            assert utter_state_list.shape == (turn_batch_num, self.max_turn, self.utterance_rnn_output_size)

            # get weight sum of utterrnn state
            # (turn_batch_num, self.utterance_rnn_output_size)
            utter_state_weight_sum = self.attention_sum(current_decoder_state, utter_state_list,
                                                        mask=turn_batch_mask)

            # (turn_batch_num, self.utterance_rnn_output_size),(turn_batch_num, self.hidden) ->(turn_batch_num, self.decoder_rnn_input_size)
            # fuse utterrnn sum and current sql embedding
            # fuse_embedding_state = self.tranform_fuse_input_state(
            #     torch.cat((utter_state_weight_sum, decoder_input_sql_embedding[:, i, :]), dim=-1))
            # we set utter rnn output + hidden == decoder rnn input, so cancal tranform_fuse_input_state module

            if self.utter_fuse:
                decoder_utter_atten_sum = self.attention_sum(current_decoder_state, turn_utter_encoder_feature,
                                                             mask=turn_utter_mask)
                fuse_embedding_state = torch.cat(
                    (utter_state_weight_sum, decoder_input_sql_embedding[:, i, :], decoder_utter_atten_sum), dim=-1)
                fuse_embedding_state = self.tranform_fuse_utter(fuse_embedding_state)
            else:
                fuse_embedding_state = torch.cat((utter_state_weight_sum, decoder_input_sql_embedding[:, i, :]), dim=-1)

            # and feed into decoder rnn
            new_decoder_state = self.decoder_rnn_cell(
                fuse_embedding_state, current_decoder_state)  # (turn_batch_num, self.decoder_rnn_output_size)

            # store decoderrnn state and update current decoder state
            if self.trigger_decode_in_out_fuse:
                decoder_state_list.append(
                    torch.tanh(
                        self.tranform_fuse_decode_in_out(torch.cat((new_decoder_state, fuse_embedding_state), dim=-1))))
            else:
                decoder_state_list.append(new_decoder_state)
            current_decoder_state = new_decoder_state
        decoder_state_list = torch.stack(decoder_state_list, dim=1)
        assert decoder_state_list.shape == (turn_batch_num, self.decode_length, self.decoder_rnn_output_size)

        return decoder_state_list

    def attention_sum(self, key, value, mask=None):
        '''
        get attention weight and sum
        :param key: turn_num-1,hidden
        :param value: turn_num-1, db_len/utter_len/sql_len, hidden
        :param mask: turn_num-1, db_len/utter_len/sql_len
        :return: turn_num-1, hidden
        '''
        attention = torch.einsum('ik,ijk -> ij', key, value)
        if mask is not None:
            assert mask.shape == attention.shape
            attention = attention.masked_fill(mask == 0, -1e9)
        weight = torch.softmax(attention, dim=-1)
        if self.hard_atten and mask is not None:
            value = value.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        weight_sum = torch.einsum('ij,ijk -> ik', weight, value)
        return weight_sum

    def split_feature(self, turn_feature):
        '''
        split three modality feature for content feature for mulitpath attention
        :param turn_feature: (turn_num - 1, self.total_len, *)
        :return: db feature, utter feature, sql feature
        '''
        return turn_feature[:, :self.db_len], turn_feature[:, self.db_len:self.db_len + self.utter_len], turn_feature[:,
                                                                                                         self.db_len + self.utter_len:],

    def feature_extractor(self, turn_embedding, turn_utter_embedding, turn_mask, turn_utter_mask):
        '''
        use transformer block to extractor feature for both content embedding and sole utterance embedding
        :param turn_embedding: #turns+1,total_len,hidden
        :param turn_utter_embedding: #turns,utter_len,hidden
        :param data:
        :return:
        '''
        if not self.pre_trans:
            turn_encoder_feature = self.tranform_layer(turn_embedding)

            turn_utter_encoder_feature = self.tranform_layer(turn_utter_embedding)

        else:

            turn_encoder_feature = turn_embedding

            turn_utter_encoder_feature = turn_utter_embedding

        if self.n_layers >= 0:
            turn_encoder_feature = self.transformer_encoder(turn_encoder_feature.permute(1, 0, 2),
                                                            src_key_padding_mask=(turn_mask == 0)).permute(1, 0, 2)
            turn_utter_encoder_feature = self.transformer_encoder(turn_utter_encoder_feature.permute(1, 0, 2),
                                                                  src_key_padding_mask=(turn_utter_mask == 0)).permute(
                1, 0, 2)

        return turn_encoder_feature, turn_utter_encoder_feature

    def extracted_db_feature(self, turn_batch_feature, turn_batch_mask=None):
        '''
        split db feature from transformer extractor,and fuse the mulit head feature, in order to get sql embedding
        :param turn_batch_feature: #turn,max_turn,len,hidden
        :param turn_batch_mask: #turn,max_turn,
        :return:# turn,db_len,hidden
        '''
        turn_batch_db_feature = turn_batch_feature[:, :, :self.db_len, :]  # turn,max_turn,db_len,hidden
        # turn_batch_mask = turn_batch_mask.unsqueeze(-1).unsqueeze(-1)
        # turn_batch_db_feature = turn_batch_db_feature.masked_fill(turn_batch_mask == 0,0.0)  # this is all 0, not need masked
        turn_batch_db_feature = turn_batch_db_feature.permute(0, 2, 1, 3)  # turn,db_len,max_turn,hidden
        # turn-1,db_len,max_turn*hidden
        if self.trigger_db_fuse_concat:
            turn_batch_db_fuse_feature = turn_batch_db_feature.reshape(turn_batch_db_feature.shape[0],
                                                                       turn_batch_db_feature.shape[1], -1)

            # turn-1,db_len,hidden
            turn_batch_db_fuse_feature = self.tranform_fuse_db_feature(turn_batch_db_fuse_feature)
        else:
            # turn-1,db_len,hidden
            turn_batch_db_fuse_feature = torch.mean(turn_batch_db_feature, dim=-2)
        if self.last_db_feature:
            turn_batch_db_fuse_feature = turn_batch_db_feature[:, :, -1, :]
        return turn_batch_db_fuse_feature

    def built_output_dbembedding(self, turn_batch_db_fuse_feature, data):
        '''
        convert turn_batch_db_fuse_feature into a embedding lookup table whose size is (# turn-1,real_db_len,hidden)
        :param turn_batch_db_fuse_feature: #(turn,db_len,hidden)
        :param data:
        :return:
        '''

        # idxs is a list contained multi continual idx
        def get_feature_from_idxs(idxs):
            # for a word group, we get and sum all word embedding to express this whole embedding
            # (turn-1,hidden)
            return torch.stack([turn_batch_db_fuse_feature[:, idx, :] for idx in idxs], dim=1).mean(dim=-2)

        def fuse_table_on_column(table_embedding, column_embedding):
            # fuse table embedding into column embedding to enhance column expression
            output, h_n = self.table_column_fuse_rnn(
                torch.stack([table_embedding, column_embedding], dim=0))  # 2*bs*hidden

            output = output.mean(dim=0)  # 2,turn-1,hidden
            # h_n = h_n.permute(1, 0, 2).reshape(h_n.shape[0], -1)  # (turn,hidden*directions)
            # return self.tranform_fuse_column_table(h_n)  # (turn,hidden)
            return output  # we set rnn state == hidden/2

        def get_embedding_strdict(column4table, content):
            current_type = 'table'
            current_table_id = 1
            idxs, idx = [], 0  # the idx for turn_batch_feature
            embedding_matrix = []
            current_table_embedding, current_table_str = None, None
            dict_list = []
            for id, token in zip(column4table, content):
                if id == 0:
                    word_group_feature = get_feature_from_idxs([id + 3 for id in idxs])
                    word_group_str = ' '.join([content[idx] for idx in idxs])
                    if current_type == 'column':
                        fuse_table_column = fuse_table_on_column(current_table_embedding, word_group_feature)
                        embedding_matrix.append(fuse_table_column)
                        dict_list.append(current_table_str + ' . ' + word_group_str)
                    elif current_type == 'table':
                        current_type = 'column'
                        current_table_embedding = word_group_feature
                        current_table_str = word_group_str
                        fuse_table_column = fuse_table_on_column(current_table_embedding, star_column)
                        embedding_matrix.append(fuse_table_column)
                        dict_list.append(current_table_str + ' . *')
                    idxs = []
                elif id == current_table_id:
                    idxs.append(idx)
                elif id != current_table_id:
                    current_type = 'table'
                    current_table_id = id
                    idxs.append(idx)
                idx += 1
            embedding_matrix.append(star_column)
            dict_list.append('. *')  # . *
            embedding_matrix.append(fuse_table_on_column(star_column, star_column))
            dict_list.append('* . *')  # * . *
            assert len(embedding_matrix) == len(dict_list)
            return embedding_matrix, dict_list

        column4table = data[0]['column4table']
        assert len(column4table) == self.db_len
        while column4table[-1] == 0: column4table = column4table[:-1]  # remove padding
        if column4table[-1] != 0: column4table.append(0)  # ensure the last one is 0 which stand for sep
        content = data[0]['content']
        star_column = get_feature_from_idxs([1])
        column4table, content = column4table[3:], content[3:]  # remove [sep] * [sep]
        embedding_matrix, dict_list = get_embedding_strdict(column4table, content)  # [(turn,hidden)*real_len]
        embedding_matrix = torch.stack(embedding_matrix, dim=1)  # (turn,db_units_num,hidden)

        return embedding_matrix, dict_list

    def lookup_from_dbembedding(self, db_embedding_matrix, db_dict_list, source_sql):
        '''
        get sql from each item of data, and convert into embedding using pre-built dbembedding matrix
        :param db_embedding_matrix: # (turn-1,db_units_num,hidden)
        :param db_dict_list: # [(str)*real_len] str is : A_1 A_2 . c_1 c_2
        :param source_sql:[['Select','From','A_1 A_2 . b_1 b_2',...]]*(turns)
        :return: decoder_source_sql_embedding : (turn_num, self.decode_length, self.hidden)
        '''

        # assert len(source_sql) == db_embedding_matrix.shape[0]
        batch_sql_embeddings = []
        for i in range(len(source_sql)):
            turn_db_embedding = db_embedding_matrix[i, :, :]
            sql_embeddings = []
            for item in source_sql[i]:

                if self.output_keyword_embedding.in_dict(item):
                    unit_embedding = self.output_keyword_embedding.convert_str_to_embedding(item).squeeze()
                else:
                    if item in db_dict_list:
                        unit_embedding = turn_db_embedding[db_dict_list.index(item), :].squeeze()
                    else:
                        print('{} is not find in DB (->replace to [* . *])'.format(item))
                        unit_embedding = turn_db_embedding[db_dict_list.index('* . *'), :].squeeze()

                sql_embeddings.append(unit_embedding)
            sql_embeddings = torch.stack(sql_embeddings, dim=0)  # unit_num,hidden
            batch_sql_embeddings.append(sql_embeddings)
        batch_sql_embeddings = torch.stack(batch_sql_embeddings, dim=0)  # turns-1,unit_num,hidden
        assert self.decode_length == batch_sql_embeddings.shape[1]  # unit_num is decode length
        return batch_sql_embeddings

    def output_prob(self, turn_batch_final_feature, db_embedding_matrix):
        '''
        transform turn_batch_final_feature to hidden size and get output prob dist (and softmax)
        :param turn_batch_final_feature: turn_batch_num, self.decode_length, self.decoder_rnn_output_size
        :param db_embedding_matrix: # (turn_batch_num,db_units_num,hidden)
        self.output_keyword_embedding: keywords * hidden
        :return:final_prob_dist #(turn_batch_num, decoder_len, keyword_num+db_unit_num)
        '''
        # turn_batch_num, self.decode_length, self.hidden
        # we set decoder_rnn_output_size == hidden, so no need to convert dim
        # turn_batch_final_feature = self.tranform_decoder_output(turn_batch_final_feature)

        # (turn_batch, decode_length, hidden) * (turn_batch,db_units_num,hidden)
        # -> turn_batch, decode_length, db_units_num

        if self.trigger_db_embedding_feature_bilinear:
            turn_batch_final_feature = torch.tanh(self.db_embedding_feature_bilinear(turn_batch_final_feature))
        db_prob_dist = torch.einsum('ijk,imk -> ijm', turn_batch_final_feature, db_embedding_matrix)
        keyword_prob_dist = self.output_keyword_embedding.convert_embedding_to_dist(turn_batch_final_feature)

        # use log_softmax not softmax
        final_prob_dist = torch.log_softmax(torch.cat((db_prob_dist, keyword_prob_dist), dim=-1), dim=-1)

        return final_prob_dist

    def caculate_loss(self, target_sql, final_prob_dist, db_dict_list):
        '''

        :param target_sql: batched target sql text sequence -> [['Select', 'A_1 A_2 . a_1 a_2', ... ]*batch]
        :param final_prob_dist: turn_batch_num, decoder_len, keyword_num+db_unit_num (db before, keyword after)
        :return: tensor
        '''

        def find_item_idx(item):
            if self.output_keyword_embedding.in_dict(item):
                return self.output_keyword_embedding.find_str_idx(item) + len(db_dict_list), 'keywords'
            else:
                if item in db_dict_list:
                    return db_dict_list.index(item), 'db_unit'
                else:
                    print('Can not find {} item of target SQL in db Dict!'.format(item))
                    return db_dict_list.index('* . *'), 'db_unit'

        total_step, valid_step, db_valid_step, key_valid_step, db_correct_step, key_correct_step = 0, 0, 0, 0, 0, 0
        assert len(target_sql) == final_prob_dist.shape[0]
        db_loss_list, key_loss_list = [], []
        total_correct_strings, total_strings = 0, 0
        for sole_sql, sole_sql_dist in zip(target_sql, final_prob_dist):
            total_strings += 1
            assert len(sole_sql) == sole_sql_dist.shape[0]
            current_sql_is_right = True
            for item, item_dist in zip(sole_sql, sole_sql_dist):
                total_step += 1
                if item == '[PAD]':
                    continue
                valid_step += 1
                idx, type = find_item_idx(item)
                if type == 'db_unit':
                    db_valid_step += 1
                    db_loss_list.append(-1 * item_dist[idx])
                    if torch.argmax(item_dist) == idx:
                        db_correct_step += 1
                    else:
                        db_correct_step += 0
                        current_sql_is_right = False

                elif type == 'keywords':
                    key_valid_step += 1
                    key_loss_list.append(-1 * item_dist[idx])
                    if torch.argmax(item_dist) == idx:
                        key_correct_step += 1
                    else:
                        key_correct_step += 0
                        current_sql_is_right = False
                # print(db_dict_list[torch.argmax(item_dist)] if torch.argmax(item_dist) < len(db_dict_list) else
                #       self.output_keyword_embedding.dict[torch.argmax(item_dist) - len(db_dict_list)])
            if current_sql_is_right:
                total_correct_strings += 1

        return {
            'db_loss': torch.sum(torch.stack(db_loss_list, dim=-1), dim=-1),
            'key_loss': torch.sum(torch.stack(key_loss_list, dim=-1), dim=-1),
            'total_step': total_step,
            'valid_step': valid_step,
            'db_valid_step': db_valid_step,
            'key_valid_step': key_valid_step,
            'db_correct_step': db_correct_step,
            'key_correct_step': key_correct_step,
            'total_strings': total_strings,
            'total_correct_strings': total_correct_strings
        }

    def create_turn_mask(self, data):
        '''

        :param data:
        :return:
        turn_mask, (turn+1,len)
        turn_utter_mask (turn,len)
        '''
        pre_turn_mask = data[0]['mask_signal'][:self.db_len] + [0] * (self.utter_len + self.sql_len)
        turn_mask = [item['mask_signal'] for item in data]
        turn_mask = torch.tensor([pre_turn_mask] + turn_mask).to(self.device)  # turns+1,total_len
        turn_utter_mask = turn_mask[1:, self.db_len:self.db_len + self.utter_len]
        return turn_mask.to(self.device), turn_utter_mask.to(self.device)

    def base_encoder(self, turn_utter_encoder_feature, turn_utter_mask, decoder_input_sql_embedding,
                     db_embedding_matrix):
        '''
        1, turn_utter_encoder_feature attn db embedding
        2, turn_utter_encoder_feature rnn
        3, decoder with attn
        :param db_embedding_matrix:(turn,db_units_num,hidden)
        :param turn_utter_encoder_feature: (turn,utter_len,hidden)
        :param turn_utter_mask:(turn,utter_len)
        :param decoder_input_sql_embedding:(turn_num, self.decode_length, self.hidden)
        :return:
        '''
        utter_len = turn_utter_encoder_feature.shape[1]
        new_turn_utter_encoder_feature = []
        for i in range(utter_len):
            new_turn_utter_encoder_feature.append(
                self.attention_sum(turn_utter_encoder_feature[:, i], db_embedding_matrix))
        new_turn_utter_encoder_feature = torch.stack(new_turn_utter_encoder_feature, dim=1)  # (turn,utter_len,hidden)
        new_turn_utter_encoder_feature = new_turn_utter_encoder_feature + turn_utter_encoder_feature
        # new_turn_utter_encoder_feature = turn_utter_encoder_feature
        if self.embedding_matrix_random: new_turn_utter_encoder_feature = turn_utter_encoder_feature
        current_decoder_state = torch.sum(
            new_turn_utter_encoder_feature.masked_fill(turn_utter_mask.unsqueeze(-1) == 0, 0.0), dim=1)
        decoder_state_list = []
        for i in range(self.decode_length):
            utter_state_weight_sum = self.attention_sum(current_decoder_state, new_turn_utter_encoder_feature,
                                                        mask=turn_utter_mask)
            fuse_embedding_state = torch.cat(
                (utter_state_weight_sum, decoder_input_sql_embedding[:, i, :]), dim=-1)
            new_decoder_state = self.decoder_rnn_cell(
                fuse_embedding_state, current_decoder_state)
            decoder_state_list.append(new_decoder_state)
            current_decoder_state = new_decoder_state
        decoder_state_list = torch.stack(decoder_state_list, dim=1)
        return decoder_state_list

    def session_loop_forward(self, data):
        # base model
        turn_embedding, turn_utter_embedding = self.mulit_modal_embedding(data)
        turn_mask, turn_utter_mask = self.create_turn_mask(data)

        turn_encoder_feature, turn_utter_encoder_feature = self.feature_extractor(turn_embedding, turn_utter_embedding,
                                                                                  turn_mask, turn_utter_mask)
        # (turn+1,len,hidd) (turn,utter_len,hidden)

        turn_num = turn_utter_encoder_feature.shape[0]
        batch_decoder_state_list = []
        batch_db_embedding_matrix = []
        target_sqls = []
        for i in range(turn_num):
            utter_feature = turn_utter_encoder_feature[i]  # (len,hidd)
            utter_mask = turn_utter_mask[i]
            history_content_feature = turn_encoder_feature[0:i + 1]  # (pre_turns,len,hidd)
            db_fuse_feature = self.extracted_db_feature(history_content_feature.unsqueeze(0))  # (1,db_len,hidd)
            db_embedding_matrix, db_dict_list = self.built_output_dbembedding(db_fuse_feature,
                                                                              data)  # (1,db_units_num,hidden)
            new_db_list = []
            for unit in db_dict_list:
                token_list = unit.split()
                while len(token_list) <= 5: token_list.append('[PAD]')
                while len(token_list) >= 5: token_list.pop()
                new_db_list.append(token_list)
            db_embedding_matrix = self.input_embedding.parse_batch_content(new_db_list, 'content')
            db_embedding_matrix = self.tranform_layer(db_embedding_matrix.mean(-2))
            batch_db_embedding_matrix.append(db_embedding_matrix.squeeze(0))
            source_sql, target_sql = data[i]['sql1'], data[i]['sql2']
            target_sqls.append(target_sql)
            decoder_input_sql_embedding = self.lookup_from_dbembedding(db_embedding_matrix.unsqueeze(0), db_dict_list,
                                                                       [source_sql])
            # (1, self.decode_length, self.hidden)
            new_utter_feature = []
            for step_feature in utter_feature:
                new_utter_feature.append(
                    self.attention_sum(step_feature.unsqueeze(0), db_embedding_matrix.unsqueeze(0)).squeeze(0))
            new_utter_feature = torch.stack(new_utter_feature, dim=0) + utter_feature

            current_decoder_state = torch.sum(new_utter_feature.masked_fill(utter_mask.unsqueeze(-1) == 0, 0.0),
                                              dim=0)  # hidden
            decoder_state_list = []
            for i in range(self.decode_length):
                utter_state_weight_sum = self.attention_sum(current_decoder_state.unsqueeze(0),
                                                            new_utter_feature.unsqueeze(0),
                                                            mask=utter_mask.unsqueeze(0)).squeeze(0)  # hidden
                fuse_embedding_state = torch.cat(
                    (utter_state_weight_sum, decoder_input_sql_embedding[0, i, :]), dim=-1)
                new_decoder_state = self.decoder_rnn_cell(
                    fuse_embedding_state.unsqueeze(0), current_decoder_state.unsqueeze(0)).squeeze(0)
                decoder_state_list.append(new_decoder_state)
                current_decoder_state = new_decoder_state
            decoder_state_list = torch.stack(decoder_state_list, dim=0)  # len,hidden
            batch_decoder_state_list.append(decoder_state_list)
        batch_decoder_state_list = torch.stack(batch_decoder_state_list, dim=0)
        batch_db_embedding_matrix = torch.stack(batch_db_embedding_matrix, dim=0)
        # (bs, decoder_len, keyword_num+db_unit_num)
        final_prob_dist = self.output_prob(batch_decoder_state_list, batch_db_embedding_matrix)
        loss_pack = self.caculate_loss(target_sqls, final_prob_dist, db_dict_list)
        return loss_pack

    def forward(self, data):
        '''
        :param data: a list of item; every item is a turn, a each one is a dict
        content -> a text list concat of db,utter,sql for this turn
        utterance -> text list included utterance for this turn
        source sql -> text list included sql for this turn, feed into decoder, ['SEP'] in first
        target sql -> text list included sql for this turn, for predict, ['SEP'] in last
        mask -> int list for transformer mask
        column4table -> int list for merge table/column expression
        :return: loss sum and loss list
        '''

        # mulit modal for content and last utterance
        turn_embedding, turn_utter_embedding = self.mulit_modal_embedding(data)
        # (turn+1,len,hidd) (turn,utt_len,hidd)

        turn_mask, turn_utter_mask = self.create_turn_mask(data)
        # (turn+1,len) (turn,utter_len)

        # use transformer block to extract feature for content and utterance
        turn_encoder_feature, turn_utter_encoder_feature = self.feature_extractor(turn_embedding, turn_utter_embedding,
                                                                                  turn_mask, turn_utter_mask)
        # (turn+1,len,hidd) (turn,utter_len,hidden)

        # split original content turn sequence into mulit session samples

        turn_batch_feature, turn_batch_mask, turn_batch_content_mask = self.create_turn_batch(turn_encoder_feature,
                                                                                              turn_mask)
        # (turn,max_turn,len,hidden) # (turn,max_turn) # (turn,max_turn,len)

        # split db feature from transformer extractor, in order to get sql embedding
        # (turn,max_turn,len,hidden) -> (turn,max_turn,db_len,hidden) -> (turn,db_len,hidden)
        turn_batch_db_fuse_feature = self.extracted_db_feature(turn_batch_feature, turn_batch_mask)

        # lookup from column4table and build db embedding and dict from turn_batch_db_fuse_feature
        db_embedding_matrix, db_dict_list = self.built_output_dbembedding(turn_batch_db_fuse_feature, data)
        # (turn,db_units_num,hidden)  [db_unit]*db_units_num

        # get source sql and target sql text sequence
        source_sql, target_sql = [item['sql1'] for item in data], [item['sql2'] for item in data]

        # new db matrix
        new_db_list = []
        for unit in db_dict_list:
            token_list = unit.split()
            while len(token_list) <= 5: token_list.append('[PAD]')
            while len(token_list) >= 5: token_list.pop()
            new_db_list.append(token_list)
        db_embedding_matrix = self.input_embedding.parse_batch_content(new_db_list, 'content')
        db_embedding_matrix = self.tranform_layer(db_embedding_matrix.mean(-2)).repeat(len(source_sql), 1,
                                                                                       1)  # 2*85*hidden

        # convert source sql into embedding using extracted db feature and keyword embedding lookup table
        decoder_input_sql_embedding = self.lookup_from_dbembedding(db_embedding_matrix, db_dict_list, source_sql)
        # (turn_num, self.decode_length, self.hidden)

        if self.embedding_matrix_random:
            # just use a randn matrix for debug
            db_embedding_matrix = torch.randn(*db_embedding_matrix.shape).to(self.device)

        if not self.base_model:
            # tow level decode for turn feature, last utterance and source sql
            turn_batch_final_feature = self.hierarchial_decode(turn_batch_feature, turn_utter_encoder_feature,
                                                               decoder_input_sql_embedding, turn_batch_mask,
                                                               turn_batch_content_mask, turn_utter_mask)
            # (turn_num, self.decode_length, self.decoder_rnn_output_size)
        else:
            turn_batch_final_feature = self.base_encoder(turn_utter_encoder_feature, turn_utter_mask,
                                                         decoder_input_sql_embedding, db_embedding_matrix)
        # convert final feature into dist, which length is (db units num + keywords num)
        # (turn_num, decoder_len, keyword_num+db_unit_num)
        final_prob_dist = self.output_prob(turn_batch_final_feature, db_embedding_matrix)
        loss_pack = self.caculate_loss(target_sql, final_prob_dist, db_dict_list)

        return loss_pack
