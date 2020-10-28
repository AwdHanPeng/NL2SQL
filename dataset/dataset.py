# TODO：创建读取SPARC和COSQL的dataset文件
import json
import pickle
import os
import re
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter
from config import opt
from data_util import ATISDataset
import torch

# SEP的temp也改成0
# data:{
# content, db_signal, temporal_signal, modality_signal, mask_signal,
# column4table(column对应元素代表其table编号)
# column2table(column对应元素代表其table位置)
# utter:
# }
class DataLoad(Dataset):
    def __init__(self, max_length, data_ori):
        self.max_length = max_length
        self.structure = []
        self.data = []
        for item in data_ori:
            self.structure.append({'database': item['split_database'], 'pair': item['split_pair']})
            self.data.append(self.get_seq(item['split_database'], item['split_pair']))

    def get_seq(self, database, pair):
        # 先将db加入序列
        db_data = ['[PAD]' for _ in range(self.max_length['db'])]
        db_db = [0 for _ in range(self.max_length['db'])]
        db_temporal = [0 for _ in range(self.max_length['db'])]
        db_modality = [0 for _ in range(self.max_length['db'])]
        db_mask = [0 for _ in range(self.max_length['db'])]
        # 附加信息：column对应元素代表其table编号
        column4table = [0 for _ in range(self.max_length['db'])]
        # 附加信息：column对应元素代表其table位置
        column2table = [0 for _ in range(self.max_length['db'])]
        table_loc = -1
        table_num = 0
        for index in range(len(database['tokens'])):
            if index + 1 >= self.max_length['db']:
                while db_data[index] != '[SEP]':
                    db_data[index] = '[PAD]'
                    db_db[index] = db_modality[index] = db_temporal[index] = 0
                    index -= 1
                break
            db_data[index] = database['tokens'][index]
            db_db[index] = database['db_signal'][index]
            db_modality[index] = database['modality_signal'][index]
            if db_data[index] != '[SEP]':
                db_mask[index] = 1
                db_temporal[index] = self.max_length['turn'] + 1
            column4table[index] = db_db[index]
            if db_db[index] > table_num:
                table_loc = index
                table_num = db_db[index]
            if db_data[index] != '[SEP]':
                column2table[index] = table_loc
            # db_temporal[index] = database['temporal_signal'][index]
        # * 的db_signal设置为最大表数+1
        db_db[1] = self.max_length['table']+1
        # 按轮数将utter和sql加入序列
        data = []
        for turn in range(len(pair)):
            # 先处理utter
            utter_data = ['[PAD]' for _ in range(self.max_length['utter'])]
            utter_db = [0 for _ in range(self.max_length['utter'])]
            utter_temporal = [0 for _ in range(self.max_length['utter'])]
            utter_modality = [0 for _ in range(self.max_length['utter'])]
            utter_mask = [0 for _ in range(self.max_length['utter'])]
            for index in range(len(pair[turn]['utter']['content'])):
                if index >= self.max_length['utter']:
                    utter_data[index-1] = '[SEP]'
                    utter_modality[index-1] = 0
                    break
                utter_data[index] = pair[turn]['utter']['content'][index]
                utter_db[index] = pair[turn]['utter']['db_signal'][index]
                utter_temporal[index] = pair[turn]['utter']['temporal_signal'][index]
                utter_modality[index] = pair[turn]['utter']['modality_signal'][index]
                if utter_data[index] != '[SEP]':
                    utter_mask[index] = 1
            # 再处理sql
            sql_data = ['[PAD]' for _ in range(self.max_length['sql'])]
            sql_db = [0 for _ in range(self.max_length['sql'])]
            sql_temporal = [0 for _ in range(self.max_length['sql'])]
            sql_modality = [0 for _ in range(self.max_length['sql'])]
            sql_mask = [0 for _ in range(self.max_length['sql'])]
            for index in range(len(pair[turn]['sql']['content'])):
                if index >= self.max_length['sql']:
                    sql_data[index - 1] = '[SEP]'
                    sql_modality[index - 1] = 0
                    break
                sql_data[index] = pair[turn]['sql']['content'][index]
                sql_db[index] = pair[turn]['sql']['db_signal'][index]
                sql_temporal[index] = pair[turn]['sql']['temporal_signal'][index]
                sql_modality[index] = pair[turn]['sql']['modality_signal'][index]
                if sql_data[index] != '[SEP]':
                    sql_mask[index] = 1
            turn_data = db_data + utter_data + sql_data
            turn_db = db_db + utter_db + sql_db
            turn_temporal = db_temporal + utter_temporal + sql_temporal
            turn_modality = db_modality + utter_modality + sql_modality
            turn_mask = db_mask + utter_mask + sql_mask
            core = {'content': turn_data, 'db_signal': turn_db, 'temporal_signal': turn_temporal, 'modality_signal': turn_modality, 'mask_signal': turn_mask}
            # 制作附加信息
            sql0 = ['[PAD]' for _ in range(self.max_length['de_sql'])]
            utter0 = ['[PAD]' for _ in range(self.max_length['de_utter'])]
            sql1 = pair[turn]['sql']['source']
            sql2 = sql1[1:]+['[SEP]']
            sql1 = [(sql1[i] if i < len(sql1) else sql0[i]) for i in range(self.max_length['de_sql'])]
            sql2 = [(sql2[i] if i < len(sql2) else sql0[i]) for i in range(self.max_length['de_sql'])]
            utter = pair[turn]['utter']['utter']
            utter = [(utter[i] if i < len(utter) else utter0[i]) for i in range(self.max_length['de_utter'])]
            core['column4table'] = column4table
            core['column2table'] = column2table
            core['utter'] = utter
            core['sql1'] = sql1
            core['sql2'] = sql2
            data.append(core)
        return data

    def __getitem__(self, index):
        assert index < len(self.data)
        return self.data[index], self.structure[index]

    def __len__(self):
        return len(self.data)


'''
:param position: -
:param modality: 0 无， 1 table 2 column 3 keyword 4 自然语言
:param temporal: 0 db， 第一轮：1 ，，，
:param db 0 无  1 table1 2 table2 3 table3 先不考虑sql
'''
# 根据ATIS构建数据集
class ATIS_DataSetLoad():
    def __init__(self, opt):
        self.qika = 'qika'
        self.opt = opt
        self.atis = ATISDataset(opt)
        self.train_ori = self.atis_leach(self.atis.train_data.examples)
        self.valid_ori = self.atis_leach(self.atis.valid_data.examples)

        # 为数据增添标识并统计最大长度
        self.max_length = self.get_length()

        # 如果自定义最大长度则覆盖
        if opt.use_max_length:
            self.max_length = opt.max_length

        # 生成数据DataSet
        self.train = DataLoad(self.max_length, self.train_ori)
        self.valid = DataLoad(self.max_length, self.valid_ori)

    # 去除数据中的多余结构
    def atis_leach(self, dataset):
        dataset_leach = []
        for item in dataset:
            leach = {}
            schema = self.read_database(item.schema.table_schema)
            leach['schema'] = item.schema
            leach['split_database'] = schema
            # leach['database_id'] = schema.table_schema['db_id']
            utterances = item.utterances
            split_interaction = []
            turn = 0
            for pair in utterances:
                turn += 1
                utterance = pair.input_seq_to_use
                sql = pair.gold_query_to_use
                split_pair = self.get_pair_type(turn, sql, utterance, item.schema)
                split_interaction.append(split_pair)
            leach['split_pair'] = split_interaction
            dataset_leach.append(leach)
        return dataset_leach

    # 填充数据类型序列
    # 返回字典{'utter':{'content','modality', 'temporal', 'db'},'sql':{'content', 'words', 'modality', 'temporal', 'db'}}
    def get_pair_type(self, turn, sql, utter, schema):
        # 计算sql
        content = []
        source = []
        modality = []
        for item in sql:
            # 当前词组出现在db中，表示为一个column词组，content需将其拆开为单词
            if item in schema.column_names_surface_form:
                table = ''
                column = item
                cut = 0
                for letter in item:
                    if letter == '.':
                        table = item[:cut]
                        column = item[(cut+1):]
                    cut += 1
                table = re.split('[ _]', table.lower())
                column_0 = re.split('[ _]', column.lower())
                column = []
                for item in column_0:
                    if item == 'departmentid':
                        column += ['department', 'id']
                    else:
                        column += [item]
                # 给出source样式
                split_word = ''
                for word in table:
                    split_word += word + ' '
                split_word += '. '
                for word in column:
                    split_word += word + ' '
                split_word = split_word[:-1]
                source.append(split_word)
                content += table + ['.'] + column
                modality += [1 for _ in table] + [3] + [2 for _ in column]
            # 当前词组是关键字
            else:
                key = re.split('[ _]', item.lower())
                split_word = ''
                for word in key:
                    split_word += word + ' '
                split_word = split_word[:-1]
                source.append(split_word)
                content += key
                modality += [3 for _ in key]
        temporal = [turn for _ in content]
        db = [0 for _ in content]
        sql = {'content': ['[SEP]'] + content + ['[SEP]'], 'modality_signal': [0] + modality + [0],
               'temporal_signal': [0] + temporal + [0], 'db_signal': [0] + db + [0],
               'source': ['[SEP]'] + source, 'target': source + ['[SEP]']}
        # 计算utter
        content = []
        source = []
        for item in utter:
            words = re.split('[ _]', item.lower())
            split_word = ''
            for word in words:
                split_word += word + ' '
            split_word = split_word[:-1]
            source.append(split_word)
            content += words
        temporal = [turn for _ in content]
        modality = [4 for _ in content]
        db = [0 for _ in content]
        utter = {'content': ['[SEP]'] + content + ['[SEP]'], 'modality_signal': [0] + modality + [0],
                 'temporal_signal': [0] + temporal + [0], 'db_signal': [0] + db + [0],
                 'utter': ['[SEP]'] + content + ['[SEP]']}
        return {'utter': utter, 'sql': sql}

    # 读取数据库并将每个数据库内表与列添加seq表示法，按[表1,[表1内的列],表2,[表2内的列],...]排列，
    # 表的表示前附加<table>符号，列的表示前附加<column>符号
    def read_database(self, table_schema):
        db_id = table_schema['db_id']
        split_database = {'id': db_id, 'tokens': [], 'db_signal': [], 'modality_signal': [], 'temporal_signal': []}
        table_num = 0
        for column in table_schema['column_names']:
            if column[0] >= table_num:
                # 加入当前一系列column对应的table
                split_table = re.split('[ _]', table_schema['table_names'][table_num].lower())
                table_num += 1
                split_database['tokens'] += ['[SEP]']
                split_database['tokens'] += split_table
                split_database['db_signal'] += [0]
                split_database['db_signal'] += [table_num for i in split_table]
                split_database['modality_signal'] += [0]
                split_database['modality_signal'] += [1 for i in split_table]
                split_database['temporal_signal'] += [0]
                split_database['temporal_signal'] += [-1 for i in split_table]
            # 处理column
            # 拆分departmentid
            split_column_0 = re.split('[ _]', column[1].lower())
            split_column = []
            for item in split_column_0:
                if item == 'departmentid':
                    split_column += ['department', 'id']
                else:
                    split_column += [item]
            split_database['tokens'] += ['[SEP]']
            split_database['tokens'] += split_column
            split_database['db_signal'] += [0]
            split_database['db_signal'] += [table_num for i in split_column]
            split_database['modality_signal'] += [0]
            split_database['modality_signal'] += [2 for i in split_column]
            split_database['temporal_signal'] += [0]
            split_database['temporal_signal'] += [-1 for i in split_column]
        # 结尾附加SEP
        split_database['tokens'] += ['[SEP]']
        split_database['db_signal'] += [0]
        split_database['modality_signal'] += [0]
        split_database['temporal_signal'] += [0]
        return split_database

    # 获取各类型最大长度
    # 数据里还有有连字符、数字、大写字母

    def get_length(self):
        max_legth = {}
        max_legth['db'] = 0
        max_legth['turn'] = 0
        max_legth['utter'] = 0
        max_legth['sql'] = 0
        max_legth['table'] = 0
        # 统计长度分布
        legth = {}
        legth['db'] = []
        legth['turn'] = []
        legth['utter'] = []
        legth['sql'] = []
        legth['table'] = []
        # 先找train最大长度
        for item in self.train_ori:
            turn = 0
            for pair in item['split_pair']:
                turn += 1
                utterance = pair['utter']['content']
                sql = pair['sql']['content']
                # 对话长度统计
                max_legth['sql'] = max(max_legth['sql'], len(sql) + 1)
                max_legth['utter'] = max(max_legth['utter'], len(utterance) + 1)
                legth['sql'].append(len(sql) + 1)
                legth['utter'].append(len(utterance) + 1)
            # 对话轮外长度统计
            max_legth['db'] = max(max_legth['db'], len(item['split_database']['tokens']))
            max_legth['turn'] = max(max_legth['turn'], turn)
            max_legth['table'] = max(max_legth['table'], len(item['schema'].table_schema['table_names']))
            legth['turn'].append(turn)
            legth['db'].append(len(item['split_database']['tokens']))
            legth['table'].append(len(item['schema'].table_schema['table_names']))

        # 再找valid最大长度
        for item in self.valid_ori:
            turn = 0
            for pair in item['split_pair']:
                turn += 1
                utterance = pair['utter']['content']
                sql = pair['sql']['content']
                # 对话长度统计
                max_legth['sql'] = max(max_legth['sql'], len(sql) + 1)
                max_legth['utter'] = max(max_legth['utter'], len(utterance) + 1)
                legth['sql'].append(len(sql) + 1)
                legth['utter'].append(len(utterance) + 1)
            # 对话轮外长度统计
            max_legth['db'] = max(max_legth['db'], len(item['split_database']['tokens']))
            max_legth['turn'] = max(max_legth['turn'], turn)
            max_legth['table'] = max(max_legth['table'], len(item['schema'].table_schema['table_names']))
            legth['turn'].append(turn)
            legth['db'].append(len(item['split_database']['tokens']))
            legth['table'].append(len(item['schema'].table_schema['table_names']))
        self.plt_length(legth, self.opt.output_root, 'turn')
        self.plt_length(legth, self.opt.output_root, 'sql')
        self.plt_length(legth, self.opt.output_root, 'utter')
        self.plt_length(legth, self.opt.output_root, 'db')
        self.plt_length(legth, self.opt.output_root, 'table')
        max_legth['de_sql'] = max_legth['sql']
        max_legth['de_utter'] = max_legth['utter']
        return max_legth

    # 统计长度
    def plt_length(self, legth, root, _type):
        length = dict(Counter(legth[_type]))
        xs = list(sorted(length.keys(), reverse=False))
        ys = [length[i] for i in xs]
        path = root + _type + '.txt'
        if os.path.exists(path):
            os.remove(path)
        file = open(path, 'w')
        sum = 0
        for fx, fy in zip(xs, ys):
            sum += fy
            file.write(str(fx))
            file.write(' ')
            file.write(str(fy))
            file.write('\n')
        file.close()
        plt.plot(xs, ys)
        plt.xlabel(_type+'_length')
        plt.ylabel('num')
        plt.legend()
        plt.savefig(root + _type+'.png')
        # plt.show()
        plt.close('all')

    # 修改最大长度
    def re_length(self, legth):
        assert ('db' in legth) and ('turn' in legth) and ('utter' in legth) and (
                'sql' in legth), "wrong keys in legth!"
        self.max_length = legth

        # 更新数据
        self.train = DataLoad(self.max_length, self.train_ori)
        self.valid = DataLoad(self.max_length, self.valid_ori)


if __name__ == '__main__':
    dataset = ATIS_DataSetLoad(opt)
    train_dataset = dataset.train
    valid_dataset = dataset.valid

