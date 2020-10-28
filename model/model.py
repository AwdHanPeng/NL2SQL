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
        self.tranform_fuse_attention = nn.Linear(self.hidden * 3, self.utterance_rnn_input_size)

        # fuse input sql embedding and utterrnn state for feed into decoder rnn
        # self.tranform_fuse_input_state = nn.Linear(self.utterance_rnn_output_size + self.hidden,
        #                                            self.decoder_rnn_input_size)

        # fuse the concated db feature into hidden dim
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
        # TODO: relu and tanh activation function and dropout have not been considerd

    def mulit_modal_embedding(self, data):
        '''
        get mulit modal embedding for content and sole utterance
        :param data: [dict{},dict{},dict{}]
        :return:
        batch_embedding, # turns,total_len,hidden
        batch_utter_embedding # turns,utter_len,hidden
        '''

        batch_content = [item['content'] for item in data]
        batch_content_embedding = self.input_embedding.parse_batch_content(batch_content,
                                                                           'content')  # turns,total_len,hidden
        batch_temporal_embedding, batch_modality_embedding, batch_db_embedding = map(
            lambda type: self.input_embedding.parse_signal(
                torch.tensor([item[type] for item in data]).to(self.device), type),
            ['temporal_signal', 'modality_signal', 'db_signal'])
        batch_positional_embedding = self.input_embedding.parse_signal(batch_content_embedding, 'position_signal')
        batch_embedding = batch_content_embedding + batch_temporal_embedding + batch_modality_embedding + batch_db_embedding + batch_positional_embedding

        batch_utterance = [item['utter'] for item in data]
        batch_utterance_embedding = self.input_embedding.parse_batch_content(batch_utterance,
                                                                             'utterance')  # turns,utter_len,hidden
        assert batch_utterance_embedding.shape[1] == self.utter_len
        batch_utter_temporal_embedding, batch_utter_modality_embedding, batch_utter_db_embedding = map(
            lambda signal, type: self.input_embedding.parse_signal(torch.tensor(signal).to(self.device), type),
            [[[i + 1] * self.utter_len for i in range(len(data))], [[4] * self.utter_len for i in range(len(data))],
             [[0] * self.utter_len for i in range(len(data))]], ['temporal_signal', 'modality_signal', 'db_signal'])

        batch_utter_positional_embedding = self.input_embedding.parse_signal(batch_utterance_embedding,
                                                                             'position_signal')
        batch_utter_embedding = batch_utterance_embedding + batch_utter_temporal_embedding + batch_utter_modality_embedding + batch_utter_db_embedding + batch_utter_positional_embedding

        return batch_embedding, batch_utter_embedding

    def create_turn_batch(self, turn_feature, turn_mask):
        '''
        convert original turn data into max-turn turn data, and the turn-1 could be viewed as batch_size
        :param turn_feature: (turn,len,hidden)
        :param turn_mask: (turn,len)
        :return:
        turn_batch_feature: (turn-1,max_turn,len,hidden);
        turn_batch_mask: (turn-1,max_turn) -> use to mask padding content for utterlevel rnn
        turn_batch_content_mask: (turn-1,max_turn,len) -> use to mask content in order to multipath
        '''
        turn_num = turn_feature.shape[0]
        turn_batch_feature = []
        turn_batch_content_mask = []
        turn_batch_mask = []
        for i in range(turn_num - 1):  # we dont need the last content pair
            left, right = (i - self.max_turn + 1) if (i - self.max_turn + 1) > 0 else 0, i
            valid_feature = turn_feature[left:right + 1, :, :]
            valid_content_mask = turn_mask[left:right + 1, :]
            if valid_feature.shape[0] < self.max_turn:
                zero_teature = torch.zeros(
                    [self.max_turn - valid_feature.shape[0], valid_feature.shape[1], valid_feature.shape[2]]).type_as(
                    valid_feature).to(self.device)
                zero_mask = torch.zeros(
                    [self.max_turn - valid_content_mask.shape[0], valid_content_mask.shape[1]]).type_as(
                    valid_content_mask).to(self.device)
                turn_batch_feature.append(torch.cat((zero_teature, valid_feature), dim=0))
                turn_batch_content_mask.append(torch.cat((zero_mask, valid_content_mask), dim=0))
                turn_batch_mask.append(torch.tensor([0] * len(zero_teature) + [1] * len(valid_feature)))

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
                           turn_batch_mask, turn_batch_content_mask, turn_utter_encoder_feature_mask):
        """
        core two level decode
        :param decoder_input_sql_embedding: (turn_num - 1, self.decode_length, self.hidden)
        :param turn_batch_feature: (turn_num - 1, self.max_turn, self.total_len, self.hidden)
        :param turn_utter_encoder_feature: (turn_num-1,self.utter_len,self.hidden)
        :param turn_batch_mask: (turn_num-1,self.max_turn)
        :param turn_batch_content_mask: (turn-1,max_turn,len) # you also need reverse
        :return: turn_num - 1, self.decode_length, self.decoder_rnn_output_size
        """
        assert decoder_input_sql_embedding.shape[1] == self.decode_length
        decoder_state_list = []
        turn_batch_num = turn_batch_feature.shape[0]

        # we get last utter sum to init the first hidden state of decoder rnn
        # current_decoder_state = torch.zeros(turn_batch_num, self.decoder_rnn_output_size)
        current_decoder_state = torch.sum(turn_utter_encoder_feature, dim=1)

        rever_turn_batch_feature = torch.flip(turn_batch_feature, dims=[1])  # reverse for utterance rnn
        rever_turn_batch_content_mask = torch.flip(turn_batch_content_mask, dims=[1])  # reverse for content mask
        for i in range(self.decode_length):
            # current_utterrnn_state = torch.zeros(turn_batch_num, self.utterance_rnn_output_size)
            # we get last utter sum to init the first hidden state of utterlevel rnn
            # current_utterrnn_state = self.tranform_lastutter_initutterrnn(torch.sum(turn_utter_encoder_feature, dim=1))
            # we set utterance_rnn_output_size == hidden, so no need to convert dim
            current_utterrnn_state = torch.sum(turn_utter_encoder_feature, dim=1)

            utterrnn_state_list = []
            for j in range(self.max_turn):
                attention_key = self.tranform_attention(
                    torch.cat((current_utterrnn_state, current_decoder_state), dim=-1))
                # get s l and atten from last utter
                last_utter_sum = self.attention_sum(attention_key, turn_utter_encoder_feature,
                                                    turn_utter_encoder_feature_mask)

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
                fuse_atten_sum = self.tranform_fuse_attention(
                    torch.cat((db_atten_sum, utter_atten_sum, sql_atten_sum),
                              dim=-1))  # turn_num-1,utterance_rnn_input_size
                # feed mulitpath attn sum into utter rnn
                new_utterrnn_state = self.utterance_rnn_cell(fuse_atten_sum)  # turn_num-1,utterance_rnn_output_size
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
            fuse_embedding_state = torch.cat((utter_state_weight_sum, decoder_input_sql_embedding[:, i, :]), dim=-1)

            # and feed into decoder rnn
            new_decoder_state = self.decoder_rnn_cell(
                fuse_embedding_state)  # (turn_batch_num, self.decoder_rnn_output_size)
            # store decoderrnn state and update current decoder state
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
        assert mask.shape == attention.shape
        if mask is not None: attention = attention.masked_fill(mask == 0, -1e9)
        weight = torch.softmax(attention, dim=-1)
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

    def feature_extractor(self, turn_embedding, turn_utter_embedding, data):
        '''
        use transformer block to extractor feature for both content embedding and sole utterance embedding
        :param turn_embedding: #turns,total_len,hidden
        :param turn_utter_embedding: #turns,utter_len,hidden
        :param data:
        :return:
        '''
        turn_mask = torch.tensor([item['mask_signal'] for item in data]).to(self.device)  # turns,total_len

        turn_encoder_feature = self.transformer_encoder(self.tranform_layer(turn_embedding.permute(1, 0, 2)),
                                                        src_key_padding_mask=(turn_mask == 0)).permute(1, 0, 2)
        turn_utter_mask = turn_mask[:, self.db_len:self.db_len + self.utter_len]
        turn_utter_encoder_feature = self.transformer_encoder(
            self.tranform_layer(turn_utter_embedding.permute(1, 0, 2)),
            src_key_padding_mask=(turn_utter_mask == 0)).permute(1, 0, 2)
        return turn_encoder_feature, turn_utter_encoder_feature

    def extracted_db_feature(self, turn_batch_feature):
        '''
        split db feature from transformer extractor,and fuse the mulit head feature, in order to get sql embedding
        :param turn_batch_feature: #turn-1,max_turn,len,hidden
        :return:# turn-1,db_len,hidden
        '''
        turn_batch_db_feature = turn_batch_feature[:, :, :self.db_len, :]  # turn-1,max_turn,db_len,hidden
        turn_batch_db_feature = turn_batch_db_feature.permute(0, 2, 1, 3)  # turn-1,db_len,max_turn,hidden
        # turn-1,db_len,max_turn*hidden
        turn_batch_db_fuse_feature = turn_batch_db_feature.reshape(turn_batch_db_feature.shape[0],
                                                                   turn_batch_db_feature.shape[1], -1)

        # turn-1,db_len,hidden
        turn_batch_db_fuse_feature = self.tranform_fuse_db_feature(turn_batch_db_fuse_feature)
        return turn_batch_db_fuse_feature

    def built_output_dbembedding(self, turn_batch_db_fuse_feature, data):
        '''
        convert turn_batch_db_fuse_feature into a embedding lookup table whose size is (# turn-1,real_db_len,hidden)
        :param turn_batch_db_fuse_feature: #(turn-1,db_len,hidden)
        :param data:
        :return:
        '''

        # idxs is a list contained multi continual idx
        def get_feature_from_idxs(idxs):
            # for a word group, we get and sum all word embedding to express this whole embedding
            # (turn-1,hidden)
            return torch.stack([turn_batch_db_fuse_feature[:, idx, :] for idx in idxs], dim=1).sum(dim=-2)

        def fuse_table_on_column(table_embedding, column_embedding):
            # fuse table embedding into column embedding to enhance column expression
            output, h_n = self.table_column_fuse_rnn(
                torch.stack([table_embedding, column_embedding], dim=0))  # 2*bs*hidden

            output = output.sum(dim=0)  # 2,turn-1,hidden
            # h_n = h_n.permute(1, 0, 2).reshape(h_n.shape[0], -1)  # (turn-1,hidden*directions)
            # return self.tranform_fuse_column_table(h_n)  # (turn-1,hidden)
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
                    word_group_feature = get_feature_from_idxs(idxs)
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
            dict_list.append(' . *')
            assert len(embedding_matrix) == len(dict_list)
            return embedding_matrix, dict_list

        column4table = data[0]['column4table']
        assert len(column4table) == self.db_len
        while column4table[-1] == 0: column4table = column4table[:-1]  # remove padding
        if column4table[-1] != 0: column4table.append(0)  # ensure the last one is 0 which stand for sep
        content = data[0]['content']
        star_column = get_feature_from_idxs([1])
        column4table, content = column4table[3:], content[3:]  # remove [sep] * [sep]
        embedding_matrix, dict_list = get_embedding_strdict(column4table, content)  # [(turn-1,hidden)*real_len]
        embedding_matrix = torch.stack(embedding_matrix, dim=1)  # (turn-1,db_units_num,hidden)

        return embedding_matrix, dict_list

    def lookup_from_dbembedding(self, db_embedding_matrix, db_dict_list, source_sql):
        '''
        get sql from each item of data, and convert into embedding using pre-built dbembedding matrix
        :param db_embedding_matrix: # (turn-1,db_units_num,hidden)
        :param db_dict_list: # [(str)*real_len] str is : A_1 A_2 . c_1 c_2
        :param source_sql:[['Select','From','A_1 A_2 . b_1 b_2',...]]*(turns-1)
        :return: decoder_source_sql_embedding : (turn_num - 1, self.decode_length, self.hidden)
        '''

        assert len(source_sql) == db_embedding_matrix.shape[0]
        batch_sql_embeddings = []
        for i in range(len(source_sql)):
            turn_db_embedding = db_embedding_matrix[i, :, :]
            sql_embeddings = []
            for item in source_sql[i]:
                if item in db_dict_list:
                    unit_embedding = turn_db_embedding[db_dict_list.index(item), :].squeeze()
                else:
                    unit_embedding = self.output_keyword_embedding.convert_str_to_embedding(item).squeeze()
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
        db_prob_dist = torch.einsum('ijk,imk -> ijm', turn_batch_final_feature, db_embedding_matrix)

        keyword_prob_dist = self.output_keyword_embedding.convert_embedding_to_dist(turn_batch_final_feature)

        final_prob_dist = torch.softmax(torch.cat((db_prob_dist, keyword_prob_dist), dim=-1), dim=-1)

        return final_prob_dist

    def caculate_loss(self, target_sql, final_prob_dist, db_dict_list):
        '''

        :param target_sql: batched target sql text sequence -> [['Select', 'A_1 A_2 . a_1 a_2', ... ]*batch]
        :param final_prob_dist: turn_batch_num, decoder_len, keyword_num+db_unit_num (db before, keyword after)
        :return: tensor
        '''

        def find_item_idx(item):
            if item in db_dict_list:
                return db_dict_list.index(item)
            else:
                return self.output_keyword_embedding.find_str_idx(item) + len(db_dict_list)

        total_step, valid_step, correct_step = 0, 0, 0
        assert len(target_sql) == final_prob_dist.shape[0]
        loss_list = []
        for sole_sql, sole_sql_dist in zip(target_sql, final_prob_dist):
            assert len(sole_sql) == sole_sql_dist.shape[0]
            for item, item_dist in zip(sole_sql, sole_sql_dist):
                total_step += 1
                if item == '[PAD]':
                    continue
                valid_step += 1
                idx = find_item_idx(item)
                loss_list.append(-1 * torch.log(item_dist[idx]))
                correct_step += (1 if torch.argmax(item_dist) == idx else 0)

        return {
            'loss': torch.sum(torch.stack(loss_list, dim=-1), dim=-1),
            'total_step': total_step,
            'valid_step': valid_step,
            'correct_step': correct_step
        }

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

        # use transformer block to extract feature for content and utterance
        turn_encoder_feature, turn_utter_encoder_feature = self.feature_extractor(turn_embedding, turn_utter_embedding,
                                                                                  data)

        # split original content turn sequence into mulit session samples
        # (turn-1,max_turn,len,hidden)
        turn_mask = torch.tensor([item['mask_signal'] for item in data]).to(self.device)
        turn_batch_feature, turn_batch_mask, turn_batch_content_mask = self.create_turn_batch(turn_encoder_feature,
                                                                                              turn_mask)

        # remove the first utterance
        # turns,utter_len,hidden -> turns-1,utter_len,hidden
        turn_utter_encoder_feature = turn_utter_encoder_feature[1:, :, :]
        turn_utter_encoder_feature_mask = turn_mask[1:, self.db_len:self.db_len + self.utter_len]

        # split db feature from transformer extractor, in order to get sql embedding
        # (turn-1,max_turn,len,hidden) -> (turn-1,max_turn,db_len,hidden) -> (turn-1,db_len,hidden)
        turn_batch_db_fuse_feature = self.extracted_db_feature(turn_batch_feature)

        # lookup from column4table and build db embedding and dict from turn_batch_db_fuse_feature
        db_embedding_matrix, db_dict_list = self.built_output_dbembedding(turn_batch_db_fuse_feature, data)
        print(db_dict_list)
        # get source sql and target sql text sequence
        source_sql, target_sql = [item['sql1'] for item in data][1:], [item['sql2'] for item in data][1:]

        # convert source sql into embedding using extracted db feature and keyword embedding lookup table
        decoder_input_sql_embedding = self.lookup_from_dbembedding(db_embedding_matrix, db_dict_list, source_sql)

        # tow level decode for turn feature, last utterance and source sql
        turn_batch_final_feature = self.hierarchial_decode(turn_batch_feature, turn_utter_encoder_feature,
                                                           decoder_input_sql_embedding, turn_batch_mask,
                                                           turn_batch_content_mask, turn_utter_encoder_feature_mask)

        # convert final feature into dist, which length is (db units num + keywords num)
        # (turn_batch_num, decoder_len, keyword_num+db_unit_num)
        final_prob_dist = self.output_prob(turn_batch_final_feature, db_embedding_matrix)
        loss_pack = self.caculate_loss(target_sql, final_prob_dist, db_dict_list)

        return loss_pack
