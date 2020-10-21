# TODO：创建读取SPARC和COSQL的dataset文件
import json
import pickle
import re
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter
from config import opt
import torch


# 将数据改为全序列形式并添加<PAD>, 单个数据形式 [columns,u1,s1,u2,s2...u_m,s_m]，s_m为希望预测获得的sql
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
        index = 0
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
            db_temporal[index] = database['temporal_signal'][index]
        db_db[0] = db_db[1] = -1
        # 按轮数将utter和sql加入序列
        data = []
        for turn in range(len(pair)):
            # 先处理utter
            utter_data = ['[PAD]' for _ in range(self.max_length['utter'])]
            utter_db = [0 for _ in range(self.max_length['utter'])]
            utter_temporal = [0 for _ in range(self.max_length['utter'])]
            utter_modality = [0 for _ in range(self.max_length['utter'])]
            for index in range(len(pair[turn]['utter']['content'])):
                if index >= self.max_length['utter']:
                    utter_data[index-1] = '[SEP]'
                    utter_modality[index-1] = 0
                    break
                utter_data[index] = pair[turn]['utter']['content'][index]
                utter_db[index] = pair[turn]['utter']['db_signal'][index]
                utter_temporal[index] = pair[turn]['utter']['temporal_signal'][index]
                utter_modality[index] = pair[turn]['utter']['modality_signal'][index]
            # 再处理sql
            sql_data = ['[PAD]' for _ in range(self.max_length['sql'])]
            sql_db = [0 for _ in range(self.max_length['sql'])]
            sql_temporal = [0 for _ in range(self.max_length['sql'])]
            sql_modality = [0 for _ in range(self.max_length['sql'])]
            for index in range(len(pair[turn]['sql']['content'])):
                if index >= self.max_length['sql']:
                    sql_data[index - 1] = '[SEP]'
                    sql_modality[index - 1] = 0
                    break
                sql_data[index] = pair[turn]['sql']['content'][index]
                sql_db[index] = pair[turn]['sql']['db_signal'][index]
                sql_temporal[index] = pair[turn]['sql']['temporal_signal'][index]
                sql_modality[index] = pair[turn]['sql']['modality_signal'][index]
            turn_data = db_data + utter_data + sql_data
            turn_db = db_db + utter_db + sql_db
            turn_temporal = db_temporal + utter_temporal + sql_temporal
            turn_modality = db_modality + utter_modality + sql_modality
            data.append({'content': turn_data, 'db_signal': turn_db, 'temporal_signal': turn_temporal, 'modality_signal': turn_modality})
        return data

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
        if opt.use_max_length:
            self.max_length = opt.max_length

        # 生成数据DataSet
        self.train = DataLoad(self.max_length, self.train_ori)
        self.dev = DataLoad(self.max_length, self.dev_ori)
        print(self.train.__getitem__(0))
        print("data already")

    # 读取数据库并将每个数据库内表与列添加seq表示法，按[表1,[表1内的列],表2,[表2内的列],...]排列，
    # 表的表示前附加<table>符号，列的表示前附加<column>符号
    def read_database(self, root):
        path = root + '/' + 'tables.json'
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
                    # 加入当前一系列column对应的table
                    split_table = re.split('[ _]', table_schema['table_names'][table_num])
                    table_num += 1
                    split_database['tokens'] += ['[SEP]']
                    split_database['tokens'] += split_table
                    split_database['db_signal'] += [table_num]
                    split_database['db_signal'] += [table_num for i in split_table]
                    split_database['modality_signal'] += [1]
                    split_database['modality_signal'] += [1 for i in split_table]
                    split_database['temporal_signal'] += [-1]
                    split_database['temporal_signal'] += [-1 for i in split_table]
                # 处理column
                split_column = re.split('[ _]', column[1])
                split_database['tokens'] += ['[SEP]']
                split_database['tokens'] += split_column
                split_database['db_signal'] += [table_num]
                split_database['db_signal'] += [table_num for i in split_column]
                split_database['modality_signal'] += [2]
                split_database['modality_signal'] += [2 for i in split_column]
                split_database['temporal_signal'] += [-1]
                split_database['temporal_signal'] += [-1 for i in split_column]
            # 结尾附加SEP
            split_database['tokens'] += ['[SEP]']
            split_database['db_signal'] += [0]
            split_database['modality_signal'] += [0]
            split_database['temporal_signal'] += [-1]
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
    # 返回字典{'utter':{'tokens','modality', 'temporal', 'db'},'sql':{'tokens', 'modality', 'temporal', 'db'}}
    def get_pair_type(self, turn, sql, utter, database):
        database_ori = self.database_schema[database['id']]
        # 计算sql
        stat = 0
        word = ''
        modality_type = []
        for item in sql:
            # 当前词未出现在db中，则表示之前一个词组已经结束，且当前词为一个关键字
            if item not in database['tokens']:
                if stat:
                    # 词组为一个表名
                    if word in database_ori['table_names']:
                        modality_type += [1 for _ in range(stat)]
                    # 词组为一个列名
                    else:
                        modality_type += [2 for _ in range(stat)]
                # 处理完一个词组，初始化，将当前关键字类型加入
                word = ''
                stat = 0
                modality_type.append(3)
            # 将当前词加入词表
            else:
                stat += 1
                word += item
        # 若结尾词组非空，记为当前词组
        if stat:
            # 词组为一个表名
            if word in database_ori['table_names']:
                modality_type += [1 for _ in range(stat)]
            # 词组为一个列名
            else:
                modality_type += [2 for _ in range(stat)]
        temporal_type = [turn for _ in sql]
        db_type = [0 for _ in sql]
        sql = {'content': ['[SEP]']+sql+['[SEP]'], 'modality_signal': [0]+modality_type+[0], 'temporal_signal': [turn]+temporal_type+[turn], 'db_signal': [0]+db_type+[0]}
        # 计算utter
        temporal_type = [turn for _ in utter]
        modality_type = [4 for _ in utter]
        db_type = [0 for _ in utter]
        utter = {'content': ['[SEP]']+utter+['[SEP]'], 'modality_signal': [0]+modality_type+[0], 'temporal_signal': [turn]+temporal_type+[turn], 'db_signal': [0]+db_type+[0]}
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

        self.plt_length(legth, 'turn')
        self.plt_length(legth, 'sql')
        self.plt_length(legth, 'utter')
        self.plt_length(legth, 'db')
        return max_legth

    # 统计长度
    def plt_length(self, legth, _type):
        length = dict(Counter(legth[_type]))
        xs = list(sorted(length.keys(), reverse=False))
        ys = [length[i] for i in xs]
        file = open(_type + '.txt', 'w')
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
        plt.savefig('data_output/'+_type+'.png')
        # plt.show()
        plt.close('all')

    # 修改最大长度
    def re_length(self, legth):
        assert ('db' in legth) and ('turn' in legth) and ('utter' in legth) and (
                'sql' in legth), "wrong keys in legth!"
        self.max_length = legth

        # 更新数据
        self.train = DataLoad(self.max_length, self.train_ori)
        self.dev = DataLoad(self.max_length, self.dev_ori)


if __name__ == '__main__':
    dataset = DataSetLoad(opt)
    train_dataset = dataset.train
    dev_dataset = dataset.dev


'''


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
<<<<<<<  utterance中每个句子的长度，如果小于args.ul padding到args.ul，如果大于args.ul那么就扔到最后的几个单词
<<<<<<<  utterance的轮数我们不处理，原来有多少轮，这个list中就放多少轮

'SQL':
['<SEP> select count ( departmentid ) from department group by departmentid <EOS> <PAD>...<PAD> ',
'<SEP> select name from department group by departmentid order by count ( departmentid ) desc limit value <EOS> <PAD>...<PAD> ']
<<<<<<<  sql中每个句子的长度，如果小于args.sl就padding到args.sl，如果大于args.sl那么就扔到最后的几个单词
<<<<<<<  sql的轮数我们不处理，原来有多少轮，这个list中就放多少轮

'input': [
concat
] 
<<<<<<<  将Table，Column，utterance1，sql1拼接起来 长度等于 tl+cl+ul+sl 的， 

    

=======
'Modality':
[[1]* tl +[2]*cl + [4]*len(utterance1) + sql1的组合}
,}
] <<<<<<<<<1表示table 2表示column 3表示utterance 4表示sql 0无

'position':
不需要，在模型forward中即可定义
’temporal‘：
不需要，在模型forward中即可定义

’dbcomponent‘:
db 0 无  1 table1 2 table2 3 table3 先不考虑sql
} 


:param position: -
:param modality: 0 无， 1 table 2 column 3 keyword 4 自然语言
:param temporal: 0 db， 第一轮：1 ，，，
:param db 0 无  1 table1 2 table2 3 table3 先不考虑sql


'''
