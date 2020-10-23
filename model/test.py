import torch

turn_batch_feature = torch.tensor(
    [[[1], [2], [3], [1], [2], [3], [3], [1], [2], [3], [1], [2], [3], [3]],
     [[4], [5], [6], [1], [2], [3], [3], [1], [2], [3], [1], [2], [3], [3]]])  # 2*6*1 (turn-1,db_len,hidden)
column2table = [1, 1, 0, 1, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0]
content = ['A_1', 'A_2', 'sep', 'b_1', 'b_2', 'sep', 'c_1', 'sep', 'B_1', 'sep', 'd_1', 'sep', 'e_1', 'sep']


def get_feature_from_idxs(idxs):
    # for a word group, we get and sum all word embedding to express this whole embedding
    # (turn-1,hidden)
    return torch.stack([turn_batch_feature[:, idx, :] for idx in idxs], dim=1).sum(dim=1)


def fuse_table_on_column(table_embedding, column_embedding):
    # fuse table embedding into column embedding to enhance column expression
    return table_embedding + column_embedding


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


print(get_embedding_strdict(column2table, content))

source_sql = ['[SEP]', 'Select', 'from', 'A1 A2 . c1 c2', ..., '[PAD]', '[PAD]']
target_sql = ['Select', 'from', 'A1 A2 . c1 c2', ..., '[SEP]', '[PAD]', '[PAD]']
