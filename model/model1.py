from .base import Base
import torch
import torch.nn as nn


class Model1(Base):
    def __init__(self, args):
        super(Model1, self).__init__(args)
        self.db_rnn = nn.GRU(self.hidden, self.decoder_rnn_output_size, batch_first=True)
        self.utter_rnn = nn.GRU(self.hidden, self.decoder_rnn_output_size, batch_first=True)

    def forward(self, data):
        db, db_mask, source_sql, target_sql, utterances, utter_masks, sqls, sql_masks = self.data_prepare(data)
        samples_num = len(source_sql)
        #  1,db_len,decoder_rnn_input_size
        db_feature, _ = self.db_rnn(self.input_embedding.parse_content(db).unsqueeze(0))

        # 1,db_units_num,hidden
        db_embedding_matrix, db_dict_list = self.built_output_dbembedding_unit(db_feature, data)
        decoder_input_sql_embedding = self.lookup_from_dbembedding(db_embedding_matrix.repeat(samples_num, 1, 1),
                                                                   db_dict_list, source_sql)
        #  bs,utter_len,,decoder_rnn_input_size
        utters_feature, _ = self.utter_rnn(self.input_embedding.parse_batch_content(utterances))

        bs_hidden_list = []
        for i in range(samples_num):
            left, right = (i - self.max_turn + 1) if (i - self.max_turn + 1) > 0 else 0, i
            utters_feature_list = utters_feature[left:right + 1].reshape(-1, self.decoder_rnn_output_size)
            utter_masks_list = utter_masks[left:right + 1].reshape(-1)

            attn_tensor = torch.cat((db_feature.squeeze(0), utters_feature_list), dim=0)  # total_len,hid
            attn_mask = torch.cat((db_mask, utter_masks_list), dim=0)  # total_len,
            current_hidden = torch.zeros((1, self.decoder_rnn_output_size))
            hidden_list = []
            for j in range(self.decode_length):
                # 1,decoder_rnn_output_size
                weight_sum = self.attention_sum(current_hidden, attn_tensor.unsqueeze(0), attn_mask.unsqueeze(0))
                current_input = torch.cat((weight_sum, decoder_input_sql_embedding[i, j].unsqueeze(0)), dim=-1)
                current_input = self.transform_output_embedding(current_input)
                new_hidden = self.decoder_rnn_cell(current_input, current_hidden)
                hidden_list.append(new_hidden)
                current_hidden = new_hidden
            hidden_list = torch.cat(hidden_list, dim=0)  # len,hidden
            bs_hidden_list.append(hidden_list)
        bs_hidden_list = torch.stack(bs_hidden_list, dim=0)  # bs,len,hidden

        final_prob_dist = self.output_prob(bs_hidden_list, db_embedding_matrix.repeat(samples_num, 1, 1), )
        loss_pack = self.calculate_loss(target_sql, final_prob_dist, db_dict_list)
        return loss_pack
