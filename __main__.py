import argparse

from torch.utils.data import DataLoader
from .dataset import ATIS_DataSetLoad as DataSetLoad
from .model import Model
from .trainer import Trainer


class DataSetConfig(object):
    def __init__(self, args):
        max_length = {
            'sql': args.sql_len,
            'utter': args.utter_len,
            'db': args.db_len,
            'turn': args.turn_num,
            'max_table': args.max_table
        }
        root = "../"
        raw_train_filename = root + "dataset/{}_data_removefrom/train.pkl".format(args.dataset)
        raw_validation_filename = root + "dataset/{}_data_removefrom/dev.pkl".format(args.dataset)
        database_schema_filename = root + "dataset/{}_data_removefrom/tables.json".format(args.dataset)
        input_vocabulary_filename = 'input_vocabulary.pkl'
        output_vocabulary_filename = 'output_vocabulary.pkl'
        data_directory = "processed_data_{}_removefrom".format(args.dataset)
        input_key = "utterance"
        processed_train_filename = 'train.pkl'
        processed_validation_filename = 'validation.pkl'
        anonymize = False
        anonymization_scoring = False
        use_snippets = False


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, type=str, default='sparc',
                        help="sparc or cosql")

    parser.add_argument("-o", "--output_path", required=True, type=str, default='output/trained_model',
                        help="model save path")

    # dataset opts
    parser.add_argument("--utter_len", type=int, default=25, help="maximum sequence length of utterance")
    parser.add_argument("--sql_len", type=int, default=65, help="maximum sequence length of sql")
    parser.add_argument("--db_len", type=int, default=300,
                        help="maximum sequence length of db (column and table)")
    parser.add_argument("--turn_num", type=int, default=7, help="maximum turn number of dialogue")
    parser.add_argument("--max_table", type=int, default=7, help="maximum table number of dialogue")
    parser.add_argument("--decoder_length", type=int, default=7, help="maximum decode step for SQL")

    # model opts
    parser.add_argument("--input_size", type=int, default=768, help="the embedding dim for content (bert dim)")
    parser.add_argument("--hidden", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("--ffn_dim", type=int, default=1024, help="ffn dim of transformer")
    parser.add_argument("--max_turn", type=int, default=4, help="valid turn of history content in a session")
    parser.add_argument("--utterrnn_input", type=int, default=512, help="utter-level rnn input size")
    parser.add_argument("--utterrnn_output", type=int, default=512,
                        help="utter-level rnn output size (equal to decoder rnn input size)")
    parser.add_argument("--decodernn_output", type=int, default=512, help="decoder rnn output size")
    parser.add_argument("--decodernn_input", type=int, default=1024, help="decoder rnn input size")
    # model opts for transformer
    parser.add_argument("--layers", type=int, default=8, help="number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")

    # trainer opts
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=200, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--load_epoch", type=int, default=-1, help="load epoch x's model param")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--lr_bert", type=float, default=3e-6, help="learning rate of adam for bert")
    parser.add_argument("--fix_bert", type=bool, default=False, help="fix bert param of fine-tune")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="warmup_steps == 4000/20")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()
    print("Loading {} Dataset".format(args.dataset))
    dataset_opt = DataSetConfig(args)
    dataset = DataSetLoad(dataset_opt)
    print("Loading Train Dataset")
    train_data_loader = dataset.train

    print("Loading test Dataset")
    test_data_loader = dataset.dev

    print("Building NL2SQL model")
    model = Model()

    print("Creating BERT Trainer")
    trainer = Trainer(model, train_dataloader=train_data_loader.data, test_dataloader=test_data_loader.data,
                      lr=args.lr, bert_lr=args.lr_bert, betas=(args.adam_beta1, args.adam_beta2),
                      weight_decay=args.adam_weight_decay, warmup_steps=args.warmup_steps,
                      with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                      bert_fix=args.fix_bert)

    if args.load_epoch >= 0:
        print("Load Epoch {}'s Param".format(args.load_epoch))
        trainer.load(epoch=args.load_epoch, file_path=args.output_path)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        valid_step_acc = trainer.test(epoch)
        trainer.save(epoch, valid_step_acc, args.output_path)
