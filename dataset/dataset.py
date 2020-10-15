# TODO：创建读取SPARC和COSQL的dataset文件
import json
import pickle
import re
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter


# 将数据改为全序列形式并添加<PAD>, 单个数据形式 [columns,u1,s1,u2,s2...u_m,s_m]
class DataLoad(Dataset):
    def __init__(self, max_length, data_ori):
        self.max_length = max_length
        self.structure = []
        self.data = []
        for item in data_ori:
            self.structure.append({'database': item['split_database'], 'pair': item['split_pair']})
            self.data.append(self.get_seq(item['split_database'], item['split_pair']))

    def get_seq(self, database, pair):
        _data = ['[PAD]' for _ in range(self.max_length['db'] + self.max_length['turn'] * (self.max_length['utter'] + self.max_length['sql']))]
        _type = [0 for _ in range(self.max_length['db'] + self.max_length['turn'] * (self.max_length['utter'] + self.max_length['sql']))]
        _temporal = _type
        _modality = _type
        for index in range(len(database['tokens'])):
            _data[index] = database['tokens'][index]
            _type[index] = database['db_signal'][index]
            _modality[index] = database['modality_signal'][index]
            _temporal[index] = database['temporal_signal'][index]
        for turn in range(len(pair)):
            for index in range(len(pair[turn]['utter']['content'])):
                _data[index + turn * (self.max_length['utter'] + self.max_length['sql']) + self.max_length['db']] \
                    = pair[turn]['utter']['content'][index]
                _type[index + turn * (self.max_length['utter'] + self.max_length['sql']) + self.max_length['db']] \
                    = pair[turn]['utter']['db_signal'][index]
                _temporal[index + turn * (self.max_length['utter'] + self.max_length['sql']) + self.max_length['db']] \
                    = pair[turn]['utter']['temporal_signal'][index]
                _modality[index + turn * (self.max_length['utter'] + self.max_length['sql']) + self.max_length['db']] \
                    = pair[turn]['utter']['modality_signal'][index]
            sql_begin = self.max_length['db'] + self.max_length['utter']
            for index in range(len(pair[turn]['sql']['content'])):
                _data[index + turn * (self.max_length['utter'] + self.max_length['sql']) + sql_begin] \
                    = pair[turn]['sql']['content'][index]
                _type[index + turn * (self.max_length['utter'] + self.max_length['sql']) + sql_begin] \
                    = pair[turn]['sql']['db_signal'][index]
                _temporal[index + turn * (self.max_length['utter'] + self.max_length['sql']) + self.max_length['db']] \
                    = pair[turn]['sql']['temporal_signal'][index]
                _modality[index + turn * (self.max_length['utter'] + self.max_length['sql']) + self.max_length['db']] \
                    = pair[turn]['sql']['modality_signal'][index]
        return {'content':_data, 'db_signal': _type, 'temporal_signal': _temporal, 'modality_signal': _modality}

    def __getitem__(self, index):
        assert index < len(self.data)
        return self.data[index]

    def __len__(self):
        return len(self.data)


# 构建数据集, 给定具体数据集，生成database，train，dev，各位置最大长度
class DataSetLoad():
    def __init__(self, opt, dataname='sparc_data', folder=''):
        self.root = folder + '/' + dataname
        if folder == '':
            self.root = dataname

        # 文件中提取
        self.database_schema, self.column_names_surface_form, self.column_names_embedder_input =\
            self.read_database(self.root)
        self.train_ori = pickle.load(open(self.root+'/'+'train.pkl', "rb+"))
        self.dev_ori = pickle.load(open(self.root+'/'+'dev.pkl', "rb+"))

        # 为数据增添标识并统计最大长度
        self.max_length = self.get_length()

        # 如果自定义最大长度则覆盖
        if opt['use_max_length']:
            self.max_length = opt['max_length']

        # 生成数据DataSet
        self.train = DataLoad(self.max_length, self.train_ori)
        self.dev = DataLoad(self.max_length, self.dev_ori)
        print(self.train.__getitem__(0))
        print("data already")

    # 读取数据库并将每个数据库内表与列添加seq表示法，按[表1,[表1内的列],表2,[表2内的列],...]排列，
    # 表的表示前附加<table>符号，列的表示前附加<column>符号
    def read_database(self, root):
        path = root+'/'+'tables.json'
        with open(path, "r") as f:
            database_schema = json.load(f)

        database_schema_dict = {}
        column_names_surface_form = []
        column_names_embedder_input = []
        for table_schema in database_schema:
            db_id = table_schema['db_id']
            split_database = {'id': db_id, 'tokens': [], 'db_signal': [], 'modality_signal': [], 'temporal_signal': []}
            table_num = 0
            for column in table_schema['column_names']:
                if column[0] >= table_num:
                    split_table = re.split('[ _]', table_schema['table_names'][table_num])
                    split_database['tokens'] += ['[SEP]']
                    split_database['tokens'] += split_table
                    split_database['db_signal'] += [0]
                    split_database['db_signal'] += [1 for i in split_table]
                    split_database['modality_signal'] += [0]
                    split_database['modality_signal'] += [1 for i in split_table]
                    split_database['temporal_signal'] += [0]
                    split_database['temporal_signal'] += [0 for i in split_table]
                    table_num += 1
                split_column = re.split('[ _]', column[1])
                split_database['tokens'] += ['[SEP]']
                split_database['tokens'] += split_column
                split_database['db_signal'] += [0]
                split_database['db_signal'] += [2 for i in split_column]
                split_database['modality_signal'] += [0]
                split_database['modality_signal'] += [1 for i in split_column]
                split_database['temporal_signal'] += [0]
                split_database['temporal_signal'] += [0 for i in split_column]
            table_schema['split_database'] = split_database
            database_schema_dict[db_id] = table_schema
            column_names = table_schema['column_names']
            column_names_original = table_schema['column_names_original']
            table_names = table_schema['table_names']
            table_names_original = table_schema['table_names_original']

            for i, (table_id, column_name) in enumerate(column_names_original):
                column_name_surface_form = column_name
                column_names_surface_form.append(column_name_surface_form.lower())

            for table_name in table_names_original:
                column_names_surface_form.append(table_name.lower())

            for i, (table_id, column_name) in enumerate(column_names):
                column_name_embedder_input = column_name
                column_names_embedder_input.append(column_name_embedder_input.split())

            for table_name in table_names:
                column_names_embedder_input.append(table_name.split())

        database = database_schema
        database_schema = database_schema_dict

        return database_schema, column_names_surface_form, column_names_embedder_input

    # 填充数据类型序列
    # 返回字典{'utter':{'tokens','db', 'temporal', 'modality'},'sql':{'tokens','db', 'temporal', 'modality'}}
    def get_pair_type(self, turn, sql, utter, database):
        database_ori = self.database_schema[database['id']]
        # 计算sql
        stat = 0
        word = ''
        db_type = []
        for item in sql:
            if item not in database['tokens']:
                if stat:
                    if word in database_ori['table_names']:
                        db_type += [1 for _ in range(stat)]
                    else:
                        db_type += [2 for _ in range(stat)]
                word = ''
                stat = 0
                db_type.append(3)
            else:
                stat += 1
                word += item
        if stat:
            if word in database_ori:
                db_type += [1 for _ in range(stat)]
            else:
                db_type += [2 for _ in range(stat)]
        temporal_type = [turn for _ in sql]
        modality_type = [3 for _ in sql]
        # 计算utter
        sql = {'content': ['[SEP]']+sql, 'db_signal': [0]+db_type, 'modality_signal': [0]+modality_type, 'temporal_signal': [0]+temporal_type}
        db_type = [0 for _ in utter]
        temporal_type = [turn for _ in utter]
        modality_type = [2 for _ in utter]
        utter = {'content': ['[SEP]']+utter, 'db_signal': [0]+db_type, 'modality_signal': [0]+modality_type, 'temporal_signal': [0]+temporal_type}
        return {'utter': utter, 'sql': sql}

    # 获取各类型最大长度，并生成分割标记好的序列化数据
    def get_length(self):
        max_legth = {}
        max_legth['db'] = 0
        max_legth['turn'] = 0
        max_legth['utter'] = 0
        max_legth['sql'] = 0
        # 统计长度分布
        legth = {}
        legth['db'] = []
        legth['turn'] = []
        legth['utter'] = []
        legth['sql'] = []
        # 先找train最大长度
        for item in self.train_ori:
            split_interaction = []
            item['split_database'] = self.database_schema[item['database_id']]['split_database']
            turn = 0
            for pair in item['interaction']:
                turn += 1
                utterance = re.split('[ _]', pair['utterance'].lower())
                sql = list(pair['sql'][0][0])
                split_pair = self.get_pair_type(turn, sql, utterance, item['split_database'])
                split_interaction.append(split_pair)
                # 对话长度统计
                max_legth['sql'] = max(max_legth['sql'], len(sql) + 1)
                max_legth['utter'] = max(max_legth['utter'], len(utterance) + 1)
                legth['sql'].append(len(sql) + 1)
                legth['utter'].append(len(utterance) + 1)
            turn += 1
            utterance = re.split('[ _]', item['final']['utterance'].lower())
            sql = re.split('[ _]', item['final']['sql'].lower())
            split_pair = self.get_pair_type(turn, sql, utterance, item['split_database'])
            split_interaction.append(split_pair)
            item['split_pair'] = split_interaction

            # 对话轮外长度统计
            max_legth['db'] = max(max_legth['db'], len(item['split_database']['tokens']))
            max_legth['sql'] = max(max_legth['sql'], len(sql) + 1)
            max_legth['utter'] = max(max_legth['utter'], len(utterance) + 1)
            max_legth['turn'] = max(max_legth['turn'], turn)
            legth['turn'].append(turn)
            legth['sql'].append(len(sql) + 1)
            legth['utter'].append(len(utterance) + 1)
            legth['db'].append(len(item['split_database']['tokens']))

        # 再找dev最大长度
        for item in self.dev_ori:
            split_interaction = []
            item['split_database'] = self.database_schema[item['database_id']]['split_database']
            turn = 0
            for pair in item['interaction']:
                turn += 1
                utterance = re.split('[ _]', pair['utterance'].lower())
                sql = list(pair['sql'][0][0])
                split_pair = self.get_pair_type(turn, sql, utterance, item['split_database'])
                split_interaction.append(split_pair)
                # 对话长度统计
                max_legth['sql'] = max(max_legth['sql'], len(sql) + 1)
                max_legth['utter'] = max(max_legth['utter'], len(utterance) + 1)
                legth['sql'].append(len(sql) + 1)
                legth['utter'].append(len(utterance) + 1)
            turn += 1
            utterance = re.split('[ _]', item['final']['utterance'].lower())
            sql = re.split('[ _]', item['final']['sql'].lower())
            split_pair = self.get_pair_type(turn, sql, utterance, item['split_database'])
            split_interaction.append(split_pair)
            item['split_pair'] = split_interaction

            # 对话轮外长度统计
            max_legth['db'] = max(max_legth['db'], len(item['split_database']['tokens']))
            max_legth['sql'] = max(max_legth['sql'], len(sql) + 1)
            max_legth['utter'] = max(max_legth['utter'], len(utterance) + 1)
            max_legth['turn'] = max(max_legth['turn'], turn)
            legth['turn'].append(turn)
            legth['sql'].append(len(sql) + 1)
            legth['utter'].append(len(utterance) + 1)
            legth['db'].append(len(item['split_database']['tokens']))

        self.plt_length(legth)
        return max_legth

    # 统计长度
    def plt_length(self, legth):
        _type = 'turn'
        length = dict(Counter(legth[_type]))

        xs = list(sorted(length.keys(), reverse=False))
        ys = [length[i] for i in xs]

        file = open(_type+'.txt', 'w')
        sum = 0
        for fx, fy in zip(xs, ys):
            sum += fy
            file.write(str(fx))
            file.write(' ')
            file.write(str(fy))
            file.write('\n')
        file.close()

        plt.plot(xs, ys)
        plt.xlabel('length')
        plt.ylabel('num')
        plt.legend()
        # plt.savefig(_type+'.png')
        plt.show()
        print(1)

    # 修改最大长度
    def re_length(self, legth):
        assert ('db' in legth) and ('turn' in legth) and ('utter' in legth) and ('sql' in legth), "wrong keys in legth!"
        self.max_length = legth

        # 更新数据
        self.train = DataLoad(self.max_length, self.train_ori)
        self.dev = DataLoad(self.max_length, self.dev_ori)


if __name__ == '__main__':
    dataset = DataSetLoad({'use_max_length': False})


# TODO：dataset的格式参考model的forward函数
