import json
import pickle
import os
import re
import wordninja
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter
from config import opt
from data_util import ATISDataset
import torch

# 加mask 类似db <SEP>utter <SEP>sql sql<SEP> (sql：组合每个column) -1改成turn+1
# 将数据改为全序列形式并添加<PAD>, 单个数据形式 [columns,u1,s1,u2,s2...u_m,s_m]，s_m为希望预测获得的sql
'''
:param position: -
:param modality: 0 无， 1 table 2 column 3 keyword 4 自然语言
:param temporal: 0 db， 第一轮：1 ，，，
:param db 0 无  1 table1 2 table2 3 table3 先不考虑sql
'''


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
        self.total_modality = self.anly_signal()
        self.total_db = self.anly_signal('db_signal')
        self.total_temporal = self.anly_signal('temporal_signal')
        self.total_mask = self.anly_signal('mask_signal')
        pass

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
        db_db[1] = self.max_length['table'] + 1

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
                    utter_data[index - 1] = '[SEP]'
                    utter_modality[index - 1] = 0
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
            core = {'content': turn_data, 'db_signal': turn_db, 'temporal_signal': turn_temporal,
                    'modality_signal': turn_modality, 'mask_signal': turn_mask}
            # 制作附加信息
            sql0 = ['[PAD]' for _ in range(self.max_length['de_sql'])]
            utter0 = ['[PAD]' for _ in range(self.max_length['de_utter'])]
            sql1 = pair[turn]['sql']['source']

            sql2 = sql1[1:] + ['[SEP]']
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

    def anly_signal(self, signal='modality_signal'):
        num = {'all': 0}
        for data in self.data:
            for pair in data:
                for p in pair[signal]:
                    num['all'] += 1
                    if p in num.keys():
                        num[p] += 1
                    else:
                        num[p] = 1
        return num

    # 把keyword加入每一条数据中，输入的keywords表为['select','and',...]形式
    def add_keywords(self, keywords):
        self.keywords = keywords
        key_data = ['[PAD]' for _ in range(self.max_length['keyword'])]
        key_db = [0 for _ in range(self.max_length['keyword'])]
        key_temporal = [0 for _ in range(self.max_length['keyword'])]
        key_modality = [0 for _ in range(self.max_length['keyword'])]
        key_mask = [0 for _ in range(self.max_length['keyword'])]
        data = []
        for key in keywords:
            data.append('[SEP]')
            data.append(key)
        data.append('[SEP]')
        legth = len(data)
        assert legth <= len(key_data), "最大长度不足以囊括所有keyword！"
        key_data = data + key_data[legth:]
        for index in range(len(data)):
            if data[index] != '[SEP]':
                key_mask[index] = 1
                key_temporal[index] = self.max_length['turn'] + 1
                key_modality[index] = 3
        for item in self.data:
            for pair in item:
                pair['content'] += key_data #+ pair['content']
                pair['db_signal'] += key_db #+ pair['db_signal']
                pair['modality_signal'] += key_modality #+ pair['modality_signal']
                pair['mask_signal'] += key_mask #+ pair['mask_signal']
                pair['temporal_signal'] += key_temporal #+ pair['temporal_signal']

    def __getitem__(self, index):
        assert index < len(self.data)
        return self.data[index], self.structure[index]

    def __len__(self):
        return len(self.data)


# 根据ATIS构建数据集
class ATIS_DataSetLoad():
    def __init__(self, opt):
        self.qika = 'qika'
        self.opt = opt
        self.keywords = opt.keywords
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

        # 附加keywords
        if opt.use_keywords:
            self.train.add_keywords(self.keywords)
            self.valid.add_keywords(self.keywords)
        pass

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
        # 将original形式的sql中db内容替换成符合自然语法的形式
        columns_ori = schema.table_schema['column_names_original']
        columns = schema.table_schema['column_names']
        tables_ori = schema.table_schema['table_names_original']
        tables = schema.table_schema['table_names']
        for item in sql:
            # 当前词组出现在db中，表示为一个column词组，content需将其拆开为单词
            if item in schema.column_names_surface_form:
                table = ''
                column = item
                cut = 0
                for letter in item:
                    if letter == '.':
                        table = item[:cut]
                        column = item[(cut + 1):]
                    cut += 1
                table = table.lower()
                column = column.lower()
                # 将original的替换成新的
                for index in range(len(tables_ori)):
                    if tables_ori[index].lower() == table and tables[index].lower() != table:
                        table = tables[index].lower()
                        break
                for index in range(len(columns_ori)):
                    table_index = columns_ori[index][0]
                    table_compare = tables[table_index].lower()
                    if columns_ori[index][1].lower() == column and columns[index][
                        1].lower() != column and table == table_compare:
                        column = columns[index][1].lower()
                        break
                table_0 = re.split('[ _]', table.lower())
                column_0 = re.split('[ _]', column.lower())
                table = []
                column = []
                for item in column_0:
                    if item == '*':
                        column += item
                    else:
                        column += wordninja.split(item)
                for item in table_0:
                    if item == '*':
                        table += item
                    else:
                        table += wordninja.split(item)
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
                if word == '*':
                    content += word
                else:
                    content += wordninja.split(word)
            split_word = split_word[:-1]
            source.append(split_word)
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
                if item == '*':
                    split_column += item
                else:
                    split_column += wordninja.split(item)
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
        # plt.plot(xs, ys)
        # plt.xlabel(_type + '_length')
        # plt.ylabel('num')
        # plt.legend()

        # plt.savefig(root + _type + '.png')

        # plt.show()
        # plt.close('all')

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
    from transformers import BertModel, BertTokenizerFast, BertConfig

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    from tqdm import tqdm

    content_list = []
    for data in train_dataset.data:
        for item in data:
            content = item['content']
            content_list += content
    for data in valid_dataset.data:
        for item in data:
            content = item['content']
            content_list += content
    word_set = set()
    for str in tqdm(content_list):
        enc = tokenizer([str], return_tensors='pt', add_special_tokens=False)
        s = enc['input_ids'].shape[1]
        if s != 1:
            word_set.add(str)
    print(word_set)
'''

q1:utter mask check 

sample:
{
    "id": "",
    "scenario": "",
    "database_id": "hospital_1",
    "interaction_id": 0,
    "final": {
        "utterance": "Find the department with the most employees.",
        "sql": "SELECT name FROM department GROUP BY departmentID ORDER BY count(departmentID) DESC LIMIT 1;"
    },
    "interaction": [
        {
            "utterance": "What is the number of employees in each department ?",
            "sql": "select count ( departmentid ) from department group by departmentid"
        },
        {
            "utterance": "Which department has the most employees ? Give me the department name .",
            "sql": "select name from department group by departmentid order by count ( departmentid ) desc limit value"
        }
    ]
}

__getitem__:

{
'Table':"<CLS> physician <SEP> department <SEP> affiliated with <SEP> procedures ....<SEP> undergoes <SEP> <PAD>...<PAD>“ <<<<<<<  小于args.tl的padding至args.tl，大于args.tl的样本需要进行截取

'Column':"<SEP> * <SEP> employee id <SEP> name <SEP> position <SEP> ... <SEP> assisting nurse <SEP> <PAD>...<PAD>" <<<<<<<  小于args.cl的padding至args.cl，大于args.cl的样本需要进行截取


‘Utterance’:
['<SEP> What is the number of employees in each department ? <SEP> <PAD>...<PAD>','<SEP> Which department has the most employees ? Give me the department name . <SEP> <PAD>...<PAD>']
'''
'''

{
'', 'Turon', 'Weirich', 'Damianfort', 'coupon', 'Hafizabad', 'pos', 'Janessa', "'Knee", 'Party_Theme', 'kayaking', 'PetroChina', '1004', 'HOU', '00:33:18', 'assignedto', "'Regular", 'Solveig', "'AKO", 'campusfee', 'emp', '8.0', 'Blume', 'fewest', "'Billund", 'fastestlapspeed', 'address_id', 'Northridge', 'allergy', 'accelerators', 'supportrepid', 'enrollments', "'Ethiopia", 'cont', 'atb', 'logon', 'hoursperweek', 'employeeid', 'nagative', "'No", 'Robbin', 'on-hold', 'Mancini', 'assistingnurse', 'LG-P760', 'aut', 'headers', "'Full", 'MPG', 'gnp', 'chervil', "'Kolob", 'mid-field', 'derparting', 'log_entry_date', "'Hawaii", "'Initial", 'Parallax', 'Rathk', 'catalogs', 'stamina', 'ppos', 'AC/DC', '50000', 'TAMI', 'roomname', "'S", 'ratingdate', 'headquarter', 'Christop', 'inidividual', 'minoring', 'surnames', 'earns', '1.85', '120000', 'alphabetic', 'IDs', 'Maudie', 'percents', 'laptime', 'Dameon', 'ppv', 'broadcasted', 'fte', 'genreid', 'invoice', 'driverstandings', '02:59:21', 'Clauss', 'collectible', 'summed', 'stageposition', 'cred', 'fun1', '102.76', '2b', 'Homenick', 'alphabetical', 'albumid', 'makeid', 'countrycode', '160000', 'q1', 'Astrids', 'Tourist_ID', "'Vermont", 'abouve', "'Published", 'to-date', 'firstname', 'percentages', 'Kayaking', '3300', "'Arizona", 'APG', 'cheapest', "'Brig", 'Dinning', 'Zinfandel', 'dorms', "'Gottlieb", '12:00:00', 'enr', 'SWEAZ', 'Abasse', 'onscholarship', 'artistid', "'Participant", "'sed", 'sporabout', 'endowments', 'gk', 'parapraph', 'Graztevski', 'stuid', 'facid', "'sint", "'Heffington", "'Joint", 'authorder', 'Spitzer', 'browswers', "'Rainbow", 'GPAs', 'attendances', 'Brander', "'American", 'arrears', "'Paid", "'Meaghan", 'incur', 'Gatwick', 'Szymon', '15000', 'interchanges', 'unitprice', 'forename', "'Friendly", '57.5', 'contaSELECT', 'useracct', 'X3', 'Derricks', '20.0', 'Abdoulaye', 'Ernser', 'A340-300', 'retailing', 'mose', "'GT", 'mib', 'mp4', 'Feil', "'Armani", '70174', 'blockfloor', 'enrolment', 'Naegeli', 'donator', 'username', 'seq', "'Grondzeiler", "'King", 'non-Catholic', 'durations', "'Smithson", 'songid', 'MADLOCK', 'dno', "'Vincent", 'fullname', 'prescribes', '300000', 'Midshipman', 'Ananthapuri', 'hight', 'billing_state', 'wifi', 'petid', 'Mergenthaler', 'Badlands', 'Katsuhiro', 'appelations', 'hse', '140000', '2192', "'Wisconsin", "'omnis", 'transcripts', 'abbreviations', 'countryname', 'Bonao', 'Porczyk', 'longest-running', 'browsers', 'LORIA', 'mailing', 'NABOZNY', 'Americano', 'hrs', '4.6', 'product_ids', "'Amisulpride", 'Chiltern', "'Morocco", 'bookings', 'Latte', 'Lohaus', 'clubdesc', "'Lynne", "'Schmidt", 'he/she', 'Peeters', 'Lockmanfurt', '10000000', 'Thiago', "'Korea", 'distributer', 'occupancy', 'ycard', 'USPS', 'Julianaside', "'Treasury", 'Motta', 'eid', 'pcp', 'FJA', 'Deckow', 'donators', 'roomid', 'Andaman', "'Marriage", 'maxoccupancy', '900.0', '94002', '15,000', 'Rebeca', "'Cancelled", 'Friesen', '18:27:16', 'indepyear', 'isava', 'teh', 'Lamberton', 'amerindian', '571', 'personfriend', 'McGranger', 'Fedex', 'tryouts', 'tolls', 'Khanewal', "'Welcome", 'billing_city', 'nicknames', 'bluetooth', 'Metallica', 'CCTV', 'HSBC', 'sumbitted', 'catnip', 'artisits', '0.99', 'Miramichi', 'roomtype', 'Vietti', 'prerequisites', 'Painless', 'synthase', 'Mariusz', '23:51:54', 'gpa', '1989-09-03', 'oth', '3452', "'intern", 'R-22', 'titils', 'lastname', 'countrylanguage', "'Electoral", 'hometowns', 'PUR', 'authorize', 'Smithson', "n't", 'dphone', "'ee", 'Leonie', 'Fearsome', 'dictinct', 'MARROTTE', 'Giuliano', 'Carribean', '2003-04-19', 'circuitid', 'Heathrow', 'studnets', 'agility', 'Geovannyton', 'fld', 'Recluse', 'pertains', 'Teqnology', '2007-12-25', 'mades', '2002-06-21', 'appointmentid', 'constructorid', "'Deleted", 'openning', 'Fami', 'HBS', 'Meaghan', 'comptroller', 'employe', 'train_number', 'ay', "'Paper", 'Ryley', 'cname', 'emailstanley.monahan', 'Bushey', 'Alloway', 'Janess', "'Orbit", 'Vat', 'statments', 'lexicographic', "'s", 'procucts', 'Melching', 'expectancy', 'descriptrion', 'locaton', 'Jone', 'enrolling', 'refund', 'gname', 'Delete', 'hbp', '30000', 'canoeing', 'code2', '5200', "'love", 'partitionid', 'porphyria', '957', 'AKW', "'ALA", 'Brenden', 'edispl', 'goalies', '180cm', 'Kertzmann', "'Olympus", 'departures', "'Alabama", 'hh', 'appellation', 'role_description', 'SWEAZY', 'appelation', 'AirCon', 'University-Channel', 'Karson', 'showtimes', 'Siudym', 'donoator', 'party_events', 'flied', 'lname', "'senior", 'wineries', 'Aerosmith', 'Erdman', 'alphabetically', 'Duplex', 'Furman', '1121', 'Jandy', 'estimations', 'CHRISSY', 'tonnage', '1976-01-01', 'mf', 'train_name', 'eg', 'Gerhold', "'North", 'checkout', 'Ph.D.', 'chargeable', '242518965', "'AIK", 'Citys', 'hanyu', "'activator", 'playlist', 'flax', "'Lake", 'appellations', 'Gruber', 'Gehling', 'dribbling', 'Keeling', 'amenid', 'VIDEOTAPE', 'gradeconversion', '_attendance', 'MySakila', 'num', "'Donceel", 'hiredate', 'CIS-220', 'clublocation', 'Fosse', 'LON', 'McEwen', 'gradepoint', 'Exp', "'Kenyatta", 'constructors', 'Beege', 'yongest', 'isofficial', 'gameid', 'bandmate', 'SELBIG', '3.8', 'shools', '10018', 'contid', 'Wnat', 'prescribe', '1.84', "'Catering", '8000', '(millions)', 'birthdate', 'work_types', 'invoices', 'Ellsworth', 'airportname', "'West", 'cmi', "'2017-06-19", 'prescriptions', 'Ottilie', '3500', 'MasterCard', "'Provisional", 'Unbound', 'mins', 'Olin', 'citizenships', 'Cobham', 'mfu', '1000000', 'sleet', "'international", 'resname', 'Wihch', 'genders', 'gtype', "'Virginia", 'tweets', 'pitstops', 'powertrain', 'player-coach', 'receipt_date', 'pommel', 'shaff', 'desc', 'Tillman', 'TMobile', 'gradepoints', 'Kohler', 'prodcuts', 'raceid', '737-800', 'titleed', 'batters', 'Jaskolski', "'Lasta", 'url', 'pettype', 'BETTE', 'src', 'bedtype', '60000', 'first-grade', 'Sawayn', 'left-handed', '636', 'invoicedate', 'example.com', "'Kelly", 'f2000', 'visibilities', 'parites', 'departmentid', 'staystart', 'authorizing', "'Dr", 'maximim', 'detentions', 'Third-rate', "'t", 'rem', "'Unsatisfied", 'Paucek', "'Sponsor", 'fnol', 'from-date', 'cumin', 'Eadie', 'paragraphs', 'Rodrick', 'MTW', 'Fujitsu', 'TARRING', "'robe", 'Anguila', "'Baldwin", 'comptrollers', 'Lubowitz', 'highschooler', 'Gleasonmouth', 'Beatrix', "'Lupe", 'gymnasts', 'Bangla', "'International", 'EVELINA', 'allergytype', 'sportsinfo', 'Zalejski', 'dpi', 'divergence', 'accout', 'q2', 'expires', 'mpg', '6862', 'sqlite', 'playl', "'1989-04-24", 'ASY', 'Fahey', 'budgeted', 'Comp', "'SF", 'Akureyi', 'toal', 'Kamila', "'Boston", 'aircrafts', '37.4', "'HMS", 'wel', "'Aripiprazole", 'lived-in', 'GOLDFINGER', 'poeple', 'Ueno', 'Lyla', 'Taizhou', '8.5', 'TRACHSEL', 'Wyman', 'climber', 'service_id', 'pct', 'dob', 'constructor', 'address2', 'mediatype', 'service_details', 'goalie', 'vat', 'alid', '2010-01-01', 'MPEG', 'forenames', 'yn', "'Close", "'Sigma", 'descriptio', "'1989-03-16", 'Feest', 'Krystel', 'dateundergoes', 'headofstate', 'bandmates', 'food-related', 'appt', 'releasedate', 'q3', 'restypename', 'KAWA', '!=', 'seatings', 'student_id', 'Moulin', 'prereq', "'Miami", 'registrations', 'baseprice', 'minit', 'lifespans', 'low_temperature', '8741', 'dname', "'Noel", 'oncall', 'Aniyah', "'Summer", 'Dr.', 'Prity', '918', 'statment', '3b', 'Bootup', 'there？', 'mkou', 'cust', 'actid', 'Goldner', 'pid', 'Koby', 'eius', 'clubname', 'occurances', 'asessment', 'shp', 'KLR209', 'datetime', 'CProxy', 'Daan', 'schooler', 'manufacte', 'Sonoma', 'market-rate', 'coasters', 'CACHEbox', 'sponser', 'body_builder', 'PHL', 'surfacearea', "'inhibitor", 'winning-pilots', 'Knolls', '07:13:53', 'Ekstraklasa', '621', 'memberships', 'Acknowledgement', 'sourceairport', 'balances', 'asc', 'edu', 'Aruba', 'Triton', 'Waterbury', 'Gorgoroth', 'statuses', 'Abilene', 'Woodroffe', 'gamesplayed', 'Turcotte', "'GV", 'furniture_id', 'duplicates', 'Monadic', 'OK，give', "'activitor", 'milliseconds', "'Aberdare", 'payed', 'divison', 'example.org', "'Jessie", 'Toure', '09:00:00', 'templates', '1986-11-13', 'AHD', "'Auto", 'right-footed', "'Express", 'Atsushi', 'arears', 'descendingly', 'right-handed', 'suppler', 'lecturers', 'Feliciaberg', 'private/public', '13,000', 'party_event', 'Rylan', 'tutors', 'postalcode', 'PU_MAN', 'custmers', "'Tabatha", "'1986-08-26", '9000', 'customer_type_code', "'Yes", 'allergies', 'QM-261', 'Guruvayur', 'confer', "'Prof", 'highshcool', 'ssn', 'Elnaugh', 'MK_MAN', '4500', '100.0', 'instructs', 'idp', 'Adivisor', 'game1', 'airportcode', "'225", "'NY", 'distinctness', 'eliminations', 'Ohh', 'adresses', 'expectancies', '563', 'perpetrator？', 'csu', 'omim', 'Despero', 'left-footed', "'Vivian", 'filename', 'fourth-grade', '8000000', 'GELL', 'flightno', 'unreviewed', "'Glenn", 'URLs', 'buildup', 'ACCT-211', 'datatypes', 'lat', 'perpetrator', '100000', 'Payam', 'Wydra', 'ht', 'horsepowers', 'apid', "'Kaloyan", 'init', 'Z520e', '900000', 'Zieme', 'jobcode', '190cm', 'Kolob', 'staffs', '42000', 'PPT', 'Tokohu', 'primary_conference', 'CTO', 'Ebba', 'tryout', "'re", 'avg', 'ExxonMobil', 'Kuhn', 'custid', "'Government", '4560596484842', '1975-01-01', 'ibb', '80000', 'coupons', 'Cleavant', 'flno', 'zipcode', 'ihsaa', '4985.0', 'Oops', "'English", 'oxen', "'Miss", 'Harford', 'Bosco', 'hanzi', 'uid', 'bname', 'ALAMO', '18000', 'piad', '94107', 'dst', 'inst', 'problem_logs', 'checkin', 'birthdays', "'Denesik", 'reflexes', 'tweet', "'Graph", 'lifeexpectancy', 'enrico09', "'Cargo", 'catalog_publisher', "'Organizer", '20000', 'Heilo', 'Nameless', 'Gelderland', 'majoring', 'wrau', "'Homeland", 'fname', "'Order", 'rentals', "'Evalyn", 'OTHA', 'laptimes', "'Matter", 'A4', 'montly', 'freinds', 'Fasterfox', 'three-year', 'crs', 'highschoolers', 'login', 'wl', 'Jolie', 'D21', 'ONDERSMA', 'St.', '2016-03-15', '94103', 'objectnumber', 'governmentform', 'Langosh', "'Monaco", 'roomnumber', 'tot', "'Tony", "'Canadian", 'Birtle', 'Falls/Grand-Sault', 'undergraduates', 'ids', "'PPT", 'socre', 'maxium', 'voluptatem', 'multiracial', '20:49:27', "'UAL", 'Cyberella', "'Soisalon", 'deparments', 'manged', 'GT-I9300', 'Lysanne', '634', 'Heaney', '12000', 'Acua', '10000', 'SSN', 'Jeramie', 'AAC', 'ATO', "'Book", 'gmail.com', 'mascots', 'postcode', 'problem_log', 'mailshot', 'interacts', "'Protoporphyrinogen", "'Private", 'billing_country', '564', 'pname', 'GPA', 'schoolers', 'Amersham', 'HKG', 'climbers', "'Tax", 'yearid', "'Tournament", '4500000', "'Robel-Schulist", 'primaryaffiliation', 'Robel', '4000000', 'NEB', 'constructorstandings', 'destairport', 'CVO', "'CA", 'countryid', 'volleys', "'Hey", 'teachest', 'county_ID', 'Blackville', 'indepdent', 'semesters', 'tweeted', 'evaluations', 'X5', 'Dayana', "'BK", 'GNP', 'fax', 'Thesisin', '200000', "'Foot", 'Ohori', 'laargest', 'amenity', 'Goodrich', 'Eau', "'Marina", 'substring', 'Sydnie', 'Rosalind', '91.0', 'blockcode', "'working", 'Farida', 'Jetblue', 'renting', '``', '2009-01-01', 'dormid', "'Tropical", 'totalenrollment', 'market_rate', 'dlocation', 'Medhurst', 'mediatypeid', 'DRE', "'University", 'Kerluke', 'driverid', "'Moving", 'shareholding', 'lf', "'Toubkal", "'APG", 'prerequisite', "'AIB", 'JPMorgan', 'detial', 'Alyson', 'overpayments', "'Lettice", '841', 'Powierza', "'Moulin", '15:06:20', 'LEIA', 'Puzzling', 'eash', 'sportname', 'Generalize', 'tourney', 'payable', '93000', '2018-03-17', 'haing', 'mailed', 'mailshots', 'WY', 'Binders', 'entry_name', 'Shieber', 'playlists', 'MOYER', 'Kapitan', '5000000', '5600', 'abbrev'}


'''
