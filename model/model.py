import torch.nn as nn

from .embedding import InputEmbedding, OutputEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size if args.input_size else None
        self.batch_size = args.batch_size if args.batch_size else None
        self.total_len = args.total_len if args.total_len else None
        self.hidden = args.hidden if args.hidden else None
        self.n_layers = args.n_layers if args.n_layers else None
        self.attn_heads = args.attn_heads if args.attn_heads else None
        self.max_turn = args.max_turn
        self.decode_length = args.decode_length
        self.utterance_rnn_input_size = args.utterrnn_input
        self.utterance_rnn_output_size = args.utterrnn_output
        self.decoder_rnn_output_size = args.decodernn_output

        # embedding for BERT, sum of several embeddings
        self.input_embedding, self.output_embedding = InputEmbedding(args), OutputEmbedding(args)

        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                      num_layers=self.n_layers)
        self.tranform_layer = nn.Linear(self.input_size, self.hidden)  # this module convert bert dim to out model dim

        self.decoder_rnn_cell = nn.GRUCell(self.utterance_rnn_output_size, self.decoder_rnn_output_size)
        self.utterance_rnn_cell = nn.GRUCell(self.utterance_rnn_input_size, self.utterance_rnn_output_size)

        # transform for concat(s,l) and then get attention weight from last utterance
        self.tranform_attention = nn.Linear(self.utterance_rnn_output_size + self.decoder_rnn_output_size, self.hidden)
        # transform for concat(s,l,last_utter_sum) and then get attention weight from every turn
        self.tranform_fuse_lastutter = nn.Linear(
            self.utterance_rnn_output_size + self.decoder_rnn_output_size + self.hidden, self.hidden)
        # fuse three modality attention weight
        self.tranform_fuse_attention = nn.Linear(self.hidden * 3, self.utterance_rnn_input_size)

        # fuse input sql embedding and utterrnn state for feed into decoder rnn
        self.tranform_fuse_input_state = nn.Linear(self.utterance_rnn_output_size + self.hidden,
                                                   self.utterance_rnn_output_size)
        # fuse the concated db feature into hidden dim
        self.tranform_fuse_db_feature = nn.Linear(self.max_turn * self.hidden, self.hidden)

        # fuse table and column to express A.b
        self.table_column_fuse_rnn = nn.GRU(self.hidden, self.hidden, bidirectional=True)
        self.tranform_fuse_column_table = nn.Linear(self.hidden * 2, self.hidden)

    def mulit_modal_embedding(self, data):
        '''
        get mulit modal embedding for content and sole utterance
        :param data: [dict{},dict{},dict{}]
        :return:
        batch_embedding, # turns,total_len,hidden
        batch_utter_embedding # turns,utter_len,hidden
        '''

        batch_content = [(' ').join(item['content']).strip() for item in data]
        batch_content_embedding = self.input_embedding.parse_batch_content(batch_content)  # turns,total_len,hidden
        batch_temporal_embedding, batch_modality_embedding, batch_db_embedding = map(
            lambda type: self.input_embedding.parse_signal(
                torch.tensor([item[type] for item in data]), type), ['temporal_signal', 'modality_signal', 'db_signal'])
        batch_positional_embedding = self.input_embedding.parse_signal(batch_content_embedding, 'position_signal')
        batch_embedding = batch_content_embedding + batch_temporal_embedding + batch_modality_embedding + batch_db_embedding + batch_positional_embedding

        batch_utterance = [(' ').join(item['utterance']).strip() for item in data]
        batch_utterance_embedding = self.input_embedding.parse_batch_content(batch_utterance)  # turns,utter_len,hidden
        assert batch_utterance_embedding.shape(1) == self.utter_len
        batch_utter_temporal_embedding, batch_utter_modality_embedding, batch_utter_db_embedding = map(
            lambda signal, type: self.input_embedding.parse_signal(torch.tensor(signal), type),
            [[[i + 1] * self.utter_len for i in range(len(data))], [[4] * self.utter_len for i in range(len(data))],
             [[0] * self.utter_len for i in range(len(data))]], ['temporal_signal', 'modality_signal', 'db_signal'])

        batch_utter_positional_embedding = self.input_embedding.parse_signal(batch_utterance_embedding,
                                                                             'position_signal')
        # TODO:we maybe have a better fuse method for text embedding and four kind signal embedding
        batch_utter_embedding = batch_utterance_embedding + batch_utter_temporal_embedding + batch_utter_modality_embedding + batch_utter_db_embedding + batch_utter_positional_embedding

        return batch_embedding, batch_utter_embedding

    def create_turn_batch(self, turn_feature):
        '''
        convert original turn data into max-turn turn data, and the turn-1 could be viewed as batch_size
        :param turn_feature: (turn,len,hidden)
        :return: (turn-1,max_turn,len,hidden)
        '''
        turn_num = turn_feature.shape[0]
        turn_batch_feature = []
        for i in range(turn_num - 1):  # we dont need the last content pair
            left, right = (i - self.max_turn + 1) if (i - self.max_turn + 1) > 0 else 0, i
            valid_feature = turn_feature[left:right + 1, :, :]
            if valid_feature.shape(0) < self.max_turn:
                zero_teature = torch.zeros(
                    [self.max_turn - valid_feature.shape(0), valid_feature.shape(1), valid_feature.shape(2)])
                turn_batch_feature.append(torch.cat((zero_teature, valid_feature), dim=0))
                # TODO,对于前边拼接零的 可以引入一个mask来防止utterance level取到结果
            else:
                turn_batch_feature.append(valid_feature)
        for item in turn_batch_feature:
            assert item.shape == (self.max_turn, self.total_len, self.hidden)
        turn_batch_feature = torch.stack(turn_batch_feature, dim=0)
        assert turn_batch_feature.shape == (turn_num - 1, self.max_turn, self.total_len, self.hidden)

        return turn_batch_feature

    def hierarchial_decode(self, turn_batch_feature, turn_utter_encoder_feature, decoder_input_sql_embedding):
        """
        core two level decode
        :param decoder_input_sql_embedding: (turn_num - 1, self.decode_length, self.hidden)
        :param turn_batch_feature: (turn_num - 1, self.max_turn, self.total_len, self.hidden)
        :param turn_utter_encoder_feature: (turn_num-1,self.utter_len,self.hidden)
        :return: turn_num - 1, self.decode_length, self.decoder_rnn_output_size
        """
        assert decoder_input_sql_embedding.shape(1) == self.decode_length
        decoder_state_list = []
        turn_batch_num = turn_batch_feature.shape(0)
        # TODO: current_decoder_state can have a better initialization
        current_decoder_state = torch.zeros(turn_batch_num, self.decoder_rnn_output_size)
        rever_turn_batch_feature = torch.flip(turn_batch_feature, dims=[1])  # reverse for utterance rnn
        for i in range(self.decode_length):
            # TODO: current_utterrnn_state can have a better initialization
            current_utterrnn_state = torch.zeros(turn_batch_num, self.utterance_rnn_output_size)
            utterrnn_state_list = []
            for j in range(self.max_turn):
                attention_key = self.tranform_attention(
                    torch.cat((current_utterrnn_state, current_decoder_state), dim=-1))
                # get s l and atten from last utter
                last_utter_sum = self.attention_sum(attention_key, turn_utter_encoder_feature)
                # fuse utter sum and s and l for atten from every turn
                fuse_attention_key = self.tranform_fuse_lastutter(
                    torch.cat((current_utterrnn_state, current_decoder_state, last_utter_sum), dim=-1))
                # split db,utter,sql feature in every turn content
                db_feature, utter_feature, sql_feature = self.split_feature(rever_turn_batch_feature[:, j, :, :])
                # get weight sum db,utter,sql feature respectively
                db_atten_sum, utter_atten_sum, sql_atten_sum = map(self.attention_sum, [fuse_attention_key] * 3,
                                                                   [db_feature, utter_feature, sql_feature])
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
            assert utter_state_list.shape == (turn_batch_num, self.max_turn, self.utterance_rnn_output_size)
            # get weight sum of utterrnn state #TODO: we can aslo use the fuse of current_utterrnn_state and last utter sum to fuse utterrnn state
            utter_state_weight_sum = self.attention_sum(current_utterrnn_state,
                                                        utterrnn_state_list)  # (turn_batch_num, self.utterance_rnn_output_size)

            # (turn_batch_num, self.utterance_rnn_output_size),(turn_batch_num, self.hidden) ->(turn_batch_num, self.decoder_rnn_input_size)
            # fuse utterrnn sum and current sql embedding
            fuse_embedding_state = self.tranform_fuse_input_state(
                torch.cat((utter_state_weight_sum, decoder_input_sql_embedding[:, i, :]), dim=-1))
            # and feed into decoder rnn
            new_decoder_state = self.decoder_rnn_cell(
                fuse_embedding_state)  # (turn_batch_num, self.decoder_rnn_output_size)
            # store decoderrnn state and update current decoder state
            decoder_state_list.append(new_decoder_state)
            current_decoder_state = new_decoder_state
            decoder_state_list = torch.stack(decoder_state_list, dim=1)
            assert decoder_state_list.shape == (turn_batch_num, self.decode_length, self.decoder_rnn_output_size)

        return decoder_state_list

    def attention_sum(self, key, value):
        '''
        get attention weight and sum
        :param key: turn_num-1,hidden
        :param value: turn_num-1, db_len/utter_len/sql_len, hidden
        :return: turn_num-1, hidden
        '''
        weight = torch.softmax(torch.einsum('ik,ijk -> ij', key, value), dim=-1)
        weight_sum = torch.einsum('ij,ijk -> ik', weight, value)
        return weight_sum

    def split_feature(self, turn_feature):
        '''
        split three modality feature for content feature for mulitpath attention
        :param turn_feature: (turn_num - 1, self.total_len, self.hidden)
        :return: db feature, utter feature, sql feature
        '''
        return turn_feature[:, :self.db_len, :], turn_feature[:, self.db_len:self.db_len + self.utter_len,
                                                 :], turn_feature[:, self.db_len + self.utter_len:, :],

    def output_prob(self, turn_batch_final_feature):
        '''

        :param turn_batch_final_feature: turn_batch_num, self.decode_length, self.decoder_rnn_output_size
        :return:
        '''

        # TODO 需要获得预表示的db特征 以及sql关键词的embedding table 并中获得概率
        pass

    def feature_extractor(self, turn_embedding, turn_utter_embedding, data):
        '''
        use transformer block to extractor feature for both content embedding and sole utterance embedding
        :param turn_embedding: #turns,total_len,hidden
        :param turn_utter_embedding: #turns,utter_len,hidden
        :param data:
        :return:
        '''
        turn_mask = torch.tensor([item['mask'] for item in data])  # turns,total_len
        turn_encoder_feature = self.transformer_encoder(self.tranform_layer(turn_embedding), mask=turn_mask)
        turn_utter_mask = turn_mask[:, self.db_len:self.db_len + self.utter_len]
        turn_utter_encoder_feature = self.transformer_encoder(self.tranform_layer(turn_utter_embedding),
                                                              mask=turn_utter_mask)
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
        turn_batch_db_fuse_feature = turn_batch_db_feature.view(turn_batch_db_feature.shape(0),
                                                                turn_batch_db_feature.shape(1), -1)

        # turn-1,db_len,hidden
        turn_batch_db_fuse_feature = self.tranform_fuse_db_feature(turn_batch_db_fuse_feature)
        return turn_batch_db_fuse_feature

    def built_output_dbembedding(self, turn_batch_feature, data):
        '''
        convert turn_batch_feature into a embedding lookup table whose size is (# turn-1,real_db_len,hidden)
        :param turn_batch_feature: #(turn-1,db_len,hidden)
        :param data:
        :return:
        '''

        # idxs is a list contained multi continual idx
        def get_feature_from_idxs(idxs):
            # for a word group, we get and sum all word embedding to express this whole embedding
            # (turn-1,hidden)
            return torch.stack([turn_batch_feature[:, idx, :] for idx in idxs], dim=1).sum(dim=1)

        def fuse_table_on_column(table_embedding, column_embedding):
            # fuse table embedding into column embedding to enhance column expression
            output, h_n = self.table_column_fuse_module(torch.stack([table_embedding, column_embedding], dim=0))
            h_n = h_n.permute(1, 0, 2).view(h_n.shape(0), -1)  # (turn-1,hidden*directions)
            return self.tranform_fuse_column_table(h_n)  # (turn-1,hidden)

        def get_embedding_strdict(column2table, content):
            current_type = 'table'
            current_table_id = 1
            idxs, idx = [], 0  # the idx for turn_batch_feature
            embedding_matrix = []
            current_table_embedding, current_table_str = None, None
            dict_list = []
            for id, token in zip(column2table, content):
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
                        embedding_matrix.append(current_table_embedding)
                        dict_list.append(current_table_str)
                    idxs = []
                elif id == current_table_id:
                    idxs.append(idx)
                elif id != current_table_id:
                    current_type = 'table'
                    current_table_id = id
                    idxs.append(idx)
                idx += 1
            assert len(embedding_matrix) == len(dict_list)
            return embedding_matrix, dict_list

        column2table = data[0]['column2table']
        assert len(column2table) == self.db_len
        while column2table[-1] == 0: column2table = column2table[:-1]  # remove padding
        if column2table[-1] != 0: column2table.append(0)  # ensure the last one is 0 which stand for sep
        content = data[0]['content']
        star_column = get_feature_from_idxs([1])
        column2table, content = column2table[3:], content[3:]  # remove [sep] * [sep]
        embedding_matrix, dict_list = get_embedding_strdict(column2table, content)  # [(turn-1,hidden)*real_len]
        # embedding_matrix = torch.stack(embedding_matrix, dim=1)
        return embedding_matrix, dict_list

    def lookup_from_dbembedding(self, db_embedding_matrix, db_dict_list, data):
        '''
        get sql from each item of data, and convert into embedding using pre-built dbembedding matrix
        :param db_embedding_matrix: # [(turn-1,hidden)*real_len]
        :param db_dict_list: # [(str)*real_len] str is : A_1 A_2 . c_1 c_2
        :param data:
        :return: decoder_input_sql_embedding : (turn_num - 1, self.decode_length, self.hidden)
        '''
        # TODO: should convert sql text into embedding
        pass

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
        '''
            data->  
            [dict{content,db_signal,temporal_signal,modality_signal},dict{},dict{}]
        
        '''

        turn_embedding, turn_utter_embedding = self.mulit_modal_embedding(data)

        # use transformer block to extract feature for content and utterance
        turn_encoder_feature, turn_utter_encoder_feature = self.feature_extractor(turn_embedding, turn_utter_embedding)

        # split original content turn sequence into mulit session samples
        turn_batch_feature = self.create_turn_batch(turn_encoder_feature)  # (turn-1,max_turn,len,hidden)

        # remove the first utterance
        turn_utter_encoder_feature = turn_utter_encoder_feature[1:, :,
                                     :]  # turns,utter_len,hidden -> turns-1,utter_len,hidden

        # TODO,拿到db预表示出来的embedidng 并初始化sql keyword embedidng，创建出outputembedding
        #  然后将input sql进行embedding表示 ？这个表示是怎么表示 直接用bert重新来一遍？ 还是取之前的多头表示
        #   并将解码的输出转换成结果

        # TODO：还没搞response的解析

        # split db feature from transformer extractor, in order to get sql embedding
        # (turn-1,max_turn,len,hidden) -> (turn-1,max_turn,db_len,hidden) -> (turn-1,db_len,hidden)
        turn_batch_db_fuse_feature = self.extracted_db_feature(turn_batch_feature)

        # lookup from column2table and build db embedding from turn_batch_db_fuse_feature
        db_embedding_matrix, db_dict_list = self.built_output_dbembedding(turn_batch_db_fuse_feature, data)

        decoder_input_sql_embedding = self.lookup_from_dbembedding(db_embedding_matrix, db_dict_list, data)

        turn_batch_final_feature = self.hierarchial_decode(turn_batch_feature, turn_utter_encoder_feature,
                                                           decoder_input_sql_embedding)

        # turn_batch_prob = self.output_embedding(turn_batch_final_feature) #turn_batch_final_feature

        # turn_batch_prob = self.output_prob(turn_batch_final_feature)

        # return turn_batch_prob
        '''
        :param position: -
        :param modality: 0 无， 1 table 2 column 3 keyword 4 自然语言
        :param temporal: 0 db， 第一轮：1 ，，，
        :param db 0 无  1 table1 2 table2 3 table3 先不考虑sql
        
        '''
