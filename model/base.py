import torch.nn as nn

from .embedding import InputEmbedding, OutputEmbedding
import torch


class Base(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.utter_len, self.db_len, self.sql_len = args.utter_len, args.db_len, args.sql_len
        self.hidden = args.hidden
        self.max_turn = args.max_turn
        self.decode_length = args.decode_length
        self.decoder_rnn_input_size = args.decodernn_input
        self.decoder_rnn_output_size = args.decodernn_output
        self.cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.hard_atten = args.hard_atten

        self.input_embedding, self.output_keyword_embedding = InputEmbedding(args), OutputEmbedding(args)
        self.decoder_rnn_cell = nn.GRUCell(self.decoder_rnn_input_size, self.decoder_rnn_output_size)
        self.table_column_fuse_rnn = nn.GRU(self.hidden, self.hidden // 2, bidirectional=True)
        self.trigger_decode_in_out_fuse = args.decode_in_out_fuse
        self.trigger_db_embedding_feature_bilinear = args.db_embedding_feature_bilinear
        if self.trigger_decode_in_out_fuse:
            self.tranform_fuse_decode_in_out = nn.Linear(self.decoder_rnn_output_size + self.decoder_rnn_input_size,
                                                         self.hidden)
        if self.trigger_db_embedding_feature_bilinear:
            self.db_embedding_feature_bilinear = nn.Linear(self.hidden, self.hidden)
        self.transform_output_embedding = nn.Linear(self.decoder_rnn_output_size + self.hidden,
                                                    self.decoder_rnn_input_size)

    def data_prepare(self, data):
        content = data[0]['content']
        db = content[:self.db_len]
        db_mask = torch.tensor(data[0]['mask_signal'][:self.db_len])
        source_sql, target_sql = [item['sql1'] for item in data], [item['sql2'] for item in data]

        utterances = [item['content'][self.db_len:self.db_len + self.utter_len] for item in data]
        utter_masks = torch.tensor([item['mask_signal'][self.db_len:self.db_len + self.utter_len] for item in data])

        sqls = [item['content'][self.db_len + self.utter_len:] for item in data]
        sql_masks = torch.tensor([item['mask_signal'][self.db_len + self.utter_len:] for item in data])
        return db, db_mask, source_sql, target_sql, utterances, utter_masks, sqls, sql_masks

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

    def output_prob(self, turn_batch_final_feature, db_embedding_matrix):
        '''
        transform turn_batch_final_feature to hidden size and get output prob dist (and softmax)
        :param turn_batch_final_feature: turn_batch_num, self.decode_length, self.decoder_rnn_output_size
        :param db_embedding_matrix: # (turn_batch_num,db_units_num,hidden)
        self.output_keyword_embedding: keywords * hidden
        :return:final_prob_dist #(turn_batch_num, decoder_len, keyword_num+db_unit_num)
        '''

        # (turn_batch, decode_length, hidden) * (turn_batch,db_units_num,hidden)
        # -> turn_batch, decode_length, db_units_num

        if self.trigger_db_embedding_feature_bilinear:
            turn_batch_final_feature = torch.tanh(self.db_embedding_feature_bilinear(turn_batch_final_feature))
        db_prob_dist = torch.einsum('ijk,imk -> ijm', turn_batch_final_feature, db_embedding_matrix)
        keyword_prob_dist = self.output_keyword_embedding.convert_embedding_to_dist(turn_batch_final_feature)

        # use log_softmax not softmax
        final_prob_dist = torch.log_softmax(torch.cat((db_prob_dist, keyword_prob_dist), dim=-1), dim=-1)

        return final_prob_dist

    def built_output_dbembedding_unit(self, turn_batch_db_fuse_feature, data):
        '''
        convert turn_batch_db_fuse_feature into a embedding lookup table whose size is (# turn-1,real_db_len,hidden)
        :param turn_batch_db_fuse_feature: #(turn,db_len,hidden)
        :param data:
        :return:
        '''

        def get_feature_from_idxs(idxs):
            # for a word group, we get and sum all word embedding to express this whole embedding
            # (turn-1,hidden)
            features = torch.stack([turn_batch_db_fuse_feature[:, idx, :] for idx in idxs], dim=1)
            return features.mean(dim=-2)

        def fuse_table_on_column(table_embedding, column_embedding):
            # fuse table embedding into column embedding to enhance column expression
            output, h_n = self.table_column_fuse_rnn(
                torch.stack([table_embedding, column_embedding], dim=0))  # 2*bs*hidden
            output = output.mean(dim=0)  # 2,turn-1,hidden
            # h_n = h_n.permute(1, 0, 2).reshape(h_n.shape[0], -1)  # (turn,hidden*directions)
            # return self.tranform_fuse_column_table(h_n)  # (turn,hidden)
            return output  # we set rnn state == hidden/2

        dict_name_list = data[0]['db_unit']
        dict_list = []
        embedding_matrix = []
        old_table_list = []
        table_feature = None
        for idx, (name, value) in enumerate(dict_name_list.items()):
            valid = True
            dict_list.append(name)
            table_list = dict_name_list[name][0]
            column_list = dict_name_list[name][1]

            for item in column_list + table_list:
                if item >= self.db_len:
                    valid = False
                    break
            if not valid: continue

            column_feature = get_feature_from_idxs(column_list)
            if len(table_list) == 0:
                fuse_table_column = fuse_table_on_column(column_feature, column_feature)
            elif old_table_list == table_list:
                fuse_table_column = fuse_table_on_column(table_feature, column_feature)
            else:
                table_feature = get_feature_from_idxs(table_list)
                old_table_list = table_list
                fuse_table_column = fuse_table_on_column(table_feature, column_feature)
            embedding_matrix.append(fuse_table_column)
            assert len(dict_list) == len(embedding_matrix)
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

    def calculate_loss(self, target_sql, final_prob_dist, db_dict_list):
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
