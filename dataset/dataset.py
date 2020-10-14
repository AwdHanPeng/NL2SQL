# TODO：创建读取SPARC和COSQL的dataset文件
import json
import pickle
import re
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class DataLoad(Dataset):
    def __init__(self, root, _type='train'):
        self.path = root + '/' + _type + '.pkl'
        self.data = pickle.load(open(self.path, "rb+"))
        self.max_length = self.getlegth()

    def getlegth(self):
        print("get_legth")



    def data2seq(self, data):
        data = []
        file_to_read = open(_path)
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            lines = lines.split()
            data.append(lines)
            pass
        return data

    def __getitem__(self, index):
        assert index < len(self.data)
        x = torch.from_numpy(self.data[index])
        # pos = torch.from_numpy(self.pos[index])
        # neg = torch.from_numpy(self.neg[index])
        return x, self.target[index]
        # return pos, neg

    def __len__(self):
        return len(self.data)

class DataSetLoad():
    # 构建数据集, 给定具体数据集，生成database，train，dev，各位置最大长度
    # 单个数据形式 [columns,u1,s1,u2,s2...u_m,s_m]
    def __init__(self, opt, dataname='sparc_data', folder=''):
        self.root = folder + '/' + dataname
        if folder == '':
            self.root = dataname
        self.database_schema, self.column_names_surface_form, self.column_names_embedder_input =\
            self.read_database(self.root)
        self.train_ori = pickle.load(open(self.root+'/'+'train.pkl', "rb+"))
        self.dev_ori = pickle.load(open(self.root+'/'+'dev.pkl', "rb+"))
        self.max_length = self.getlegth()
        print(0)

    def read_database(self, root):
        path = root+'/'+'tables.json'
        with open(path, "r") as f:
            database_schema = json.load(f)

        database_schema_dict = {}
        column_names_surface_form = []
        column_names_embedder_input = []
        for table_schema in database_schema:
            db_id = table_schema['db_id']
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
    def getlegth(self):
        # 现在还没统计db
        max_legth = {}
        max_legth['db'] = 0
        max_legth['turn'] = 0
        max_legth['utter'] = 0
        max_legth['sql'] = 0
        # 先找train最大长度
        for item in self.train_ori:
            split_interaction = []
            turn = len(item['interaction'])
            max_legth['turn'] = max(max_legth['turn'], turn + 1)
            for pair in item['interaction']:
                utterance = re.split('[ _]', pair['utterance'])
                sql = pair['sql']
                max_legth['sql'] = max(max_legth['sql'], len(sql))
                max_legth['utter'] = max(max_legth['utter'], len(utterance))
                split_pair = {'utterance': utterance, 'sql': sql}
                split_interaction.append(split_pair)
            utterance = re.split('[ _]', item['final']['utterance'])
            sql = item['final']['sql']
            max_legth['sql'] = max(max_legth['sql'], len(sql))
            max_legth['utter'] = max(max_legth['utter'], len(utterance))
            item['split_interaction'] = split_interaction
            item['split_final'] = {'utterance': utterance, 'sql': sql}

        # 再找dev最大长度
        for item in self.dev_ori:
            split_interaction = []
            turn = len(item['interaction'])
            max_legth['turn'] = max(max_legth['turn'], turn + 1)
            for pair in item['interaction']:
                utterance = re.split('[ _]', pair['utterance'])
                sql = pair['sql']
                max_legth['sql'] = max(max_legth['sql'], len(sql))
                max_legth['utter'] = max(max_legth['utter'], len(utterance))
                split_pair = {'utterance': utterance, 'sql': sql}
                split_interaction.append(split_pair)
            utterance = re.split('[ _]', item['final']['utterance'])
            sql = item['final']['sql']
            max_legth['sql'] = max(max_legth['sql'], len(sql))
            max_legth['utter'] = max(max_legth['utter'], len(utterance))
            item['split_interaction'] = split_interaction
            item['split_final'] = {'utterance': utterance, 'sql': sql}

dataset = DataSetLoad(1)








# TODO：dataset的格式参考model的forward函数
