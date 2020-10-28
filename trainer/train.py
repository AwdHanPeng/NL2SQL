import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .optim_schedule import ScheduledOptim

import tqdm


class Trainer:

    def __init__(self, model,
                 train_dataloader, test_dataloader, bert_fix,
                 lr: float = 1e-3, bert_lr: float = 1e-5, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 warmup_steps=200,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, ):
        """
        :param  :   model which you want to train
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for   training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This model will be saved every epoch
        self.model = model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.params, self.params_bert = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'model_bert' in name:
                    self.params_bert.append(param)
                else:
                    self.params.append(param)
        self.optim = Adam(self.params, lr=lr, betas=betas, weight_decay=weight_decay)

        self.bert_optim = Adam(self.params_bert, lr=bert_lr, betas=betas,
                               weight_decay=weight_decay) if not bert_fix else None

        self.optim_schedule = ScheduledOptim(self.optim, self.model.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)  # idx 0 not have a loss

        self.log_freq = log_freq
        self.best_valid_acc = float('-inf')
        self.best_epoch = 0

        print('=====================Parameters in Optimizer==============')
        for param_group in self.model.optim.param_groups:
            print(param_group.keys())
            for param in param_group['params']:
                print(param.size())

        print("Total Parameters: {}*1e6", sum([p.nelement() for p in self.params]) // 1e6)

    def train(self, epoch):

        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: step average acc
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:

            # 1. forward the input and all position labels
            generated_sql = self.model(data)

            loss_pack = self.criterion(generated_sql, data["gold sql"])

            loss, total_step, valid_step, correct_step = loss_pack['loss'], loss_pack['total_step'], loss_pack[
                'valid_step'], loss_pack['correct_step']

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                if self.bert_optim: self.bert_optim.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
                if self.bert_optim: self.bert_optim.step()

            avg_loss += loss.item()
            total_correct += correct_step
            total_element += valid_step

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "step_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "step_acc=",
              total_correct * 100.0 / total_element)
        return total_correct / total_element

    def save(self, epoch, step_acc, file_path):
        """
        Saving the current  model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        # TODO：目前只实现了teacher force的acc计算过程 迭代式尚未实现
        if step_acc >= self.best_valid_acc:
            self.best_valid_acc = step_acc
            self.best_epoch = epoch
            output_path = file_path + ".ep%d.pth" % epoch
            torch.save(self.model.state_dict(), output_path)
            self.model.to(self.device)
            print("EP:%d Model Saved on:" % epoch, output_path)
            print("Current Valid Acc is {} in {} epoch" % step_acc, epoch)
            return output_path
        else:
            print("EP:%d Model NO Save" % epoch)
            print("Best Valid Acc is {} in {} epoch" % self.best_valid_acc, self.best_epoch)
            print("Current Valid Acc is {} in {} epoch" % step_acc, epoch)

    def load(self, epoch, file_path):
        output_path = file_path + ".ep%d.pth" % epoch
        self.model.load_state_dict(torch.load(output_path))
