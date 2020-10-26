import warnings


class DefaultConfig(object):
    seed = 3435
    num_epochs = 100
    use_gpu = True  # user GPU or not
    gpu_id = 0
    use_max_length = True
    max_length = {
        'sql': 65,
        'utter': 25,
        'db': 300,
        'turn': 7
    }
    root = "../"
    raw_train_filename = root + "dataset/sparc_data_removefrom/train.pkl"
    raw_validation_filename = root + "dataset/sparc_data_removefrom/dev.pkl"
    database_schema_filename = root + "dataset/sparc_data_removefrom/tables.json"
    # embedding_filename ="/home/lily/rz268/dialog2sql/word_emb/glove.840B.300d.txt"
    input_vocabulary_filename = 'input_vocabulary.pkl'
    output_vocabulary_filename = 'output_vocabulary.pkl'
    data_directory = "processed_data_sparc_removefrom"
    input_key = "utterance"
    processed_train_filename = 'train.pkl'
    processed_validation_filename = 'validation.pkl'
    anonymize = False
    anonymization_scoring = False
    use_snippets = False



def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        # 更新配置参数
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


DefaultConfig.parse = parse
opt = DefaultConfig()
print(opt)