import argparse

from torch.utils.data import DataLoader
from dataset import ATIS_DataSetLoad as DataSetLoad
from model import Model
from trainer import Trainer
import os
import pickle


class DataSetConfig(object):
    def __init__(self, args):
        self.use_keywords = True
        self.keywords_ori = ['=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group by', 'order by',
                     'distinct', 'and', 'limit value', 'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<',
                     'sum', 'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!=', 'union', 'between', '-',
                     '+', '/']
        self.keywords = ['=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group by', 'order by',
                     'distinct', 'and', 'limit value', 'limit', 'descend', '>', 'average', 'have', 'max', 'in', '<',
                     'sum', 'intersect', 'not', 'min', 'except', 'or', 'ascend', 'like', '! =', 'union', 'between', '-',
                     '+', '/']
        self.max_length = {
            'sql': args.sql_len,
            'utter': args.utter_len,
            'db': args.db_len,
            'turn': args.turn_num,
            'table': args.max_table,
            'keyword': 75,
            'de_utter': args.utter_len,
            'de_sql': args.decode_length
        }
        self.root = "./"
        self.output_root = self.root + "dataset/data_output/"
        self.raw_train_filename = self.root + "dataset/{}_data_removefrom/train.pkl".format(args.dataset)
        self.raw_validation_filename = self.root + "dataset/{}_data_removefrom/dev.pkl".format(args.dataset)
        self.database_schema_filename = self.root + "dataset/{}_data_removefrom/tables.json".format(args.dataset)
        self.input_vocabulary_filename = 'input_vocabulary.pkl'
        self.output_vocabulary_filename = 'output_vocabulary.pkl'
        self.data_directory = "processed_data_{}_removefrom".format(args.dataset)
        self.input_key = "utterance"
        self.processed_train_filename = 'train.pkl'
        self.processed_validation_filename = 'validation.pkl'
        self.anonymize = False
        self.anonymization_scoring = False
        self.use_snippets = False
        self.use_max_length = True


class DefaultConfig(object):
    seed = 3435
    num_epochs = 100
    use_gpu = True  # user GPU or not
    gpu_id = 0
    use_max_length = True
    use_keywords = True
    keywords_ori = ['=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group by', 'order by',
                     'distinct', 'and', 'limit value', 'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<',
                     'sum', 'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!=', 'union', 'between', '-',
                     '+', '/']
    keywords = ['=', 'select', 'value', ')', '(', 'where', ',', 'count', 'group by', 'order by',
                     'distinct', 'and', 'limit value', 'limit', 'descend', '>', 'average', 'have', 'max', 'in', '<',
                     'sum', 'intersect', 'not', 'min', 'except', 'or', 'ascend', 'like', '! =', 'union', 'between', '-',
                     '+', '/']
    max_length = {
        'sql': 65,
        'utter': 30,
        'db': 300,
        'turn': 6,
        'table': 26,
        'keyword': 75,
        'de_sql': 65,
        'de_utter': 30
    }
    root = "F:/NL2SQL"
    output_root = root + "dataset/data_output/"
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


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, default='sparc', help="sparc or cosql")

    parser.add_argument("-o", "--output_path", type=str, default='output/trained_model', help="model save path")
    parser.add_argument("--dataset_path", type=str, default='./catch/', help="dataset save path")
    parser.add_argument("--save_bert_vocab", type=bool, default=True, help="Save bert vocab or not")

    # dataset opts
    parser.add_argument("--utter_len", type=int, default=40, help="maximum sequence length of utterance")
    parser.add_argument("--sql_len", type=int, default=50, help="maximum sequence length of sql")
    parser.add_argument("--db_len", type=int, default=422,
                        help="maximum sequence length of db (column and table)")
    parser.add_argument("--turn_num", type=int, default=6, help="maximum turn number of dialogue")
    parser.add_argument("--max_table", type=int, default=26, help="maximum table number of dialogue")
    parser.add_argument("--decode_length", type=int, default=65, help="maximum decode step for SQL")
    parser.add_argument("--shuffle", type=bool, default=False,
                        help="shuffle the train dataset")

    # model opts
    parser.add_argument("--input_size", type=int, default=768, help="the embedding dim for content (bert dim)")
    parser.add_argument("--hidden", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("--ffn_dim", type=int, default=1024, help="ffn dim of transformer")
    parser.add_argument("--max_turn", type=int, default=5, help="valid turn of history content in a session")
    parser.add_argument("--utterrnn_input", type=int, default=512, help="utter-level rnn input size")
    parser.add_argument("--utterrnn_output", type=int, default=512,
                        help="utter-level rnn output size (equal to decoder rnn input size)")
    parser.add_argument("--decodernn_output", type=int, default=512, help="decoder rnn output size")
    parser.add_argument("--decodernn_input", type=int, default=1024, help="decoder rnn input size")
    # model opts for transformer
    parser.add_argument("--n_layers", type=int, default=8, help="number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")

    # trainer opts
    parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=20, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--load_epoch", type=int, default=-1, help="load epoch x's model param")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--use_bert", type=bool, default=True, help="use bert or not")
    parser.add_argument("--lr_bert", type=float, default=1e-5, help="learning rate of adam for bert")
    parser.add_argument("--fix_bert", type=bool, default=False, help="fix bert param of fine-tune")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="warmup_steps == 4000/20")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    # model enhance
    parser.add_argument("--decode_in_out_fuse", type=bool, default=True, help="fuse decoder output and input")
    parser.add_argument("--db_embedding_feature_bilinear", type=bool, default=True,
                        help="bilinear between decoder feature and db embedding")
    parser.add_argument("--db_fuse_concat", type=bool, default=False,
                        help="fuse mulit db feature use concat or add")

    # model debug
    parser.add_argument("--tiny_dataset", type=bool, default=False, help="use 10 sample to debug")
    parser.add_argument("--warmup", type=bool, default=False, help="warmup or not")
    parser.add_argument("--grad_clip", type=bool, default=False, help="grad clip or not")
    parser.add_argument("--hard_atten", type=bool, default=True, help="Avoid [0]*N mask, still get a sum")
    parser.add_argument("--pre_trans", type=bool, default=True,
                        help="True: pre convert bert/glove embedding into dim and then add signal; False: add signal and then convert dim")
    parser.add_argument("--key_feature_init", type=bool, default=False, help="init key embedding use content feature")
    parser.add_argument("--key_file_init", type=bool, default=False, help="read embedding file to init key embedding")
    parser.add_argument("--utter_fuse", type=bool, default=True, help="fuse utter during decode step")
    parser.add_argument("--three_fuse", type=bool, default=True, help="fuse utter and sql, except db in decode step")
    parser.add_argument("--base_model", type=bool, default=True, help="base model")
    parser.add_argument("--use_signal", type=bool, default=True, help="use siganl or not")
    parser.add_argument("--embedding_matrix_random", type=bool, default=False, help="use siganl or not")
    parser.add_argument("--last_db_feature", type=bool, default=False, help="just use one turn's db feature")
    parser.add_argument("--session_loop", type=bool, default=False, help="session loop for debug")
    args = parser.parse_args()

    print("Loading {} Dataset".format(args.dataset))

    dataset_path = args.dataset_path + args.dataset + '.pkl'

    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    else:

        dataset_opt = DataSetConfig(args)
        dataset = DataSetLoad(dataset_opt)
        with open(dataset_path, 'wb') as f:
            print('Save dataset.pkl in {}'.format(dataset_path))
            pickle.dump(dataset, f)

    print("Loading Train Dataset")
    train_data_loader = dataset.train.data
    print("Loading Test Dataset")
    test_data_loader = dataset.valid.data

    if args.tiny_dataset:
        print('Use Tiny Dateset')
        train_data_loader = train_data_loader[:5]
        test_data_loader = None
        # test_data_loader = test_data_loader[:5]

    print("Building NL2SQL model")
    model = Model(args)

    # download bert
    print("Creating BERT Trainer")
    trainer = Trainer(model=model, train_dataloader=train_data_loader, test_dataloader=test_data_loader, args=args)

    if args.load_epoch >= 0:
        print("Load Epoch {}'s Param".format(args.load_epoch))
        trainer.load(epoch=args.load_epoch, file_path=args.output_path, )

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        if test_data_loader is not None:
            valid_step_acc, valid_string_acc = trainer.test(epoch)
            trainer.save(epoch, valid_step_acc, valid_string_acc, args.output_path)


if __name__ == '__main__':
    train()
