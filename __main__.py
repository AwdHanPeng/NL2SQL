import argparse

from torch.utils.data import DataLoader
from .dataset import DataSetLoad
from .model import Model
from .trainer import Trainer


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="model save path")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-ul", "--utter_len", type=int, default=20, help="maximum sequence length of utterance")
    parser.add_argument("-sl", "--sql_len", type=int, default=20, help="maximum sequence length of sql")
    parser.add_argument("-dl", "--db_len", type=int, default=20,
                        help="maximum sequence length of db (column and table)")

    parser.add_argument("-tn", "--turn", type=int, default=20, help="maximum turn number of dialogue")

    # parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")  # lazy or full

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()
    print("Loading Dataset", args.train_dataset)
    dataset = DataSetLoad(args)
    print("Loading Train Dataset", args.train_dataset)
    train_data_loader = dataset.train

    print("Loading test Dataset", args.test_dataset)
    test_data_loader = dataset.dev if args.test_dataset is not None else None

    print("Building NL2SQL model")
    model = Model()

    print("Creating BERT Trainer")
    trainer = Trainer(model, train_dataloader=train_data_loader.data, test_dataloader=test_data_loader.data,
                      lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                      with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)
        # FIXME：保存模型未根据指标进行选择性的存储模型，需要进一步更改，且未实现early stop
        # FIXME: 测试时并未使用beam search等方法进行迭代生成，而仅测试了测试数据集上的tearcher forcing的acc
        if test_data_loader is not None:
            trainer.test(epoch)
