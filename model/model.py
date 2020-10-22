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
        # embedding for BERT, sum of several embeddings
        self.input_embedding, self.output_embedding = InputEmbedding(args), OutputEmbedding(args)

        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                      num_layers=self.n_layers)
        self.tranform_layer = nn.Linear(self.input_size, self.hidden)  # this module convert bert dim to out model dim

        self.utterance_rnn_input_size = args.utterrnn_input
        self.utterance_rnn_output_size = args.utterrnn_output
        self.decoder_rnn_output_size = args.decodernn_output
        self.decoder_rnn_cell = nn.GRUCell(self.utterance_rnn_output_size, self.decoder_rnn_output_size)
        self.utterance_rnn_cell = nn.GRUCell(self.utterance_rnn_input_size, self.utterance_rnn_output_size)

        # transform for concat(s,l) and then get attention weight
        self.tranform_attention = nn.Linear(self.utterance_rnn_output_size + self.decoder_rnn_output_size, self.hidden)
        # fuse three modality attention weight
        self.tranform_fuse_attention = nn.Linear(self.hidden * 3, self.utterance_rnn_input_size)

        # fuse input sql embedding and rnn decoder state
        self.tranform_fuse_input_state = nn.Linear(self.decoder_rnn_output_size + self.hidden,
                                                   self.utterance_rnn_input_size)

    def create_mask(self, signal):
        assert signal.size == (self.batch_size, self.total_len)
        return signal.unsqueeze(1).repeat(1, signal.size(1), 1).unsqueeze(1)

    def mulit_modal_embedding(self, data):
        batch_content = [(' ').join(item['content']).strip() for item in data]
        batch_content_embedding = self.input_embedding.parse_batch_content(batch_content)
        # TODO，输入到bert和transformer里边都没引进mask
        batch_temporal_embedding, batch_modality_embedding, batch_db_embedding = map(
            lambda type: self.input_embedding.parse_signal(
                torch.tensor([item[type] for item in data]), type), ['temporal_signal', 'modality_signal', 'db_signal'])

        return batch_content_embedding + batch_temporal_embedding + batch_modality_embedding + batch_db_embedding

    def create_turn_batch(self, turn_feature):
        '''
        convert original turn data into max-turn turn data
        :param turn_feature: (turn,len,hidden)
        :return: (turn-1,max_turn,len,hidden)
        '''
        turn_num = turn_feature.shape[0]
        turn_batch_feature = []
        for i in range(turn_num - 1):
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

    def hierarchial_decode(self, decoder_input_sql, turn_batch_feature):
        """
        :param decoder_input_sql: (turn_num - 1, self.decode_length, self.hidden)
        :param turn_batch_feature: (turn_num - 1, self.max_turn, self.total_len, self.hidden)
        :return: turn_num - 1, self.decode_length, self.decoder_rnn_output_size
        """
        assert decoder_input_sql.shape(1) == self.decode_length
        decoder_state_list = []
        turn_batch_num = turn_batch_feature.shape(0)
        current_decoder_state = torch.zeros(turn_batch_num, self.decoder_rnn_output_size)
        for i in range(self.decode_length):
            current_utterrnn_state = torch.zeros(turn_batch_num, self.utterance_rnn_output_size)
            utterrnn_state_list = []
            for j in range(self.max_turn):
                attention_key = self.tranform_attention(
                    torch.cat((current_utterrnn_state, current_decoder_state), dim=-1))
                db_feature, utter_feature, sql_feature = self.split_feature(turn_batch_feature[:, j, :, :])

                db_atten_sum, utter_atten_sum, sql_atten_sum = map(self.attention_sum, [attention_key] * 3,
                                                                   [db_feature, utter_feature, sql_feature])
                fuse_atten_sum = self.tranform_fuse_attention(
                    torch.cat((db_atten_sum, utter_atten_sum, sql_atten_sum),
                              dim=-1))  # turn_num-1,utterance_rnn_input_size
                new_utterrnn_state = self.utterance_rnn_cell(fuse_atten_sum)  # turn_num-1,utterance_rnn_output_size
                utterrnn_state_list.append(new_utterrnn_state)
                current_utterrnn_state = new_utterrnn_state

            utter_state_list = torch.stack(utterrnn_state_list, dim=1)
            assert utter_state_list.shape == (turn_batch_num, self.max_turn, self.utterance_rnn_output_size)
            utter_state_weight_sum = self.attention_sum(current_utterrnn_state,
                                                        utterrnn_state_list)  # (turn_batch_num, self.utterance_rnn_output_size)

            # (turn_batch_num, self.utterance_rnn_output_size),(turn_batch_num, self.hidden) ->(turn_batch_num, self.decoder_rnn_input_size)
            fuse_embedding_state = self.tranform_fuse_input_state(
                torch.cat((utter_state_weight_sum, decoder_input_sql[:, i, :]), dim=-1))

            new_decoder_state = self.decoder_rnn_cell(
                fuse_embedding_state)  # (turn_batch_num, self.decoder_rnn_output_size)
            decoder_state_list.append(new_decoder_state)
            current_decoder_state = new_decoder_state
            decoder_state_list = torch.stack(decoder_state_list, dim=1)
            assert decoder_state_list.shape == (turn_batch_num, self.decode_length, self.decoder_rnn_output_size)

        return decoder_state_list

    def attention_sum(self, key, value):
        '''

        :param key: turn_num-1,hidden
        :param value: turn_num-1, db_len/utter_len/sql_len, hidden
        :return: turn_num-1, hidden
        '''
        weight = torch.softmax(torch.einsum('ik,ijk -> ij', key, value), dim=-1)
        weight_sum = torch.einsum('ij,ijk -> ik', weight, value)
        return weight_sum

    def split_feature(self, turn_feature):
        '''
        :param turn_feature: (turn_num - 1, self.total_len, self.hidden)
        :return:
        '''
        return turn_feature[:, :self.db_len, :], turn_feature[:, self.db_len + 1:self.db_len + self.utter_len,
                                                 :], turn_feature[:, self.db_len + self.utter_len + 1:, :],

    def output_prob(self, turn_batch_final_feature):
        '''

        :param turn_batch_final_feature: turn_batch_num, self.decode_length, self.decoder_rnn_output_size
        :return:
        '''

        # TODO 需要获得预表示的db特征 以及sql关键词的embedding table 并中获得概率
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

        turn_embedding = self.mulit_modal_embedding(data)

        turn_encoder_feature = self.transformer_encoder_layer(self.tranform_layer(turn_embedding))

        turn_batch_feature = self.create_turn_batch(turn_encoder_feature)

        # TODO,拿到db预表示出来的embedidng 并初始化sql keyword embedidng，创建出outputembedding
        #  然后将input sql进行embedding表示 ？这个表示是怎么表示 直接用bert重新来一遍？ 还是取之前的多头表示
        #   并将解码的输出转换成结果


        #TODO：还没搞response的解析
        turn_batch_final_feature = self.hierarchial_decode(decoder_input_sql, turn_batch_feature)

        # turn_batch_prob = self.output_embedding(turn_batch_final_feature) #turn_batch_final_feature

        turn_batch_prob = self.output_prob(turn_batch_final_feature)

        return turn_batch_prob
        '''
        :param position: -
        :param modality: 0 无， 1 table 2 column 3 keyword 4 自然语言
        :param temporal: 0 db， 第一轮：1 ，，，
        :param db 0 无  1 table1 2 table2 3 table3 先不考虑sql
        
        '''
