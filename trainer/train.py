import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .optim_schedule import ScheduledOptim

import tqdm


class Trainer:

    def __init__(self, model, args,
                 train_dataloader, test_dataloader=None, ):
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

        self.log_freq = args.log_freq
        weight_decay = args.adam_weight_decay
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.model = model.to(self.device)
        if args.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.params, self.params_name, self.params_bert, self.params_bert_name = [], [], [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bert' in name:
                    self.params_bert.append(param)
                    self.params_bert_name.append(name)
                else:
                    self.params.append(param)
                    self.params_name.append(name)
        betas = (args.adam_beta1, args.adam_beta2)
        self.optim = Adam(self.params, lr=args.lr, betas=betas, weight_decay=weight_decay)
        self.warmup = args.warmup
        self.bert_optim = Adam(self.params_bert, lr=args.lr_bert, betas=betas,
                               weight_decay=weight_decay) if (args.use_bert and not args.fix_bert) else None

        if self.warmup:
            self.optim_schedule = ScheduledOptim(self.optim, self.model.hidden, n_warmup_steps=args.warmup_steps)

        self.best_valid_acc = float('-inf')
        self.best_epoch = 0
        self.tensorboard_writer = SummaryWriter()
        self.iter = -1
        print('=====================Module in Optimizer==============')
        for name, param in zip(self.params_name, self.params):
            print('Module {}: {}*1e3'.format(name, sum([p.nelement() for p in param]) // 1e3))
        print('=====================Module in Bert==============')
        for name, param in zip(self.params_bert_name, self.params_bert):
            print('Module {}: {}*1e3'.format(name, sum([p.nelement() for p in param]) // 1e3))
        print("Total Parameters: {}*1e6".format(sum([p.nelement() for p in self.params]) // 1e6))
        print("Total Bert Parameters: {}*1e6".format(sum([p.nelement() for p in self.params_bert]) // 1e6))
        self.shuffle = args.shuffle
        self.grad_clip = args.grad_clip
        self.session_loop = args.session_loop

    def train(self, epoch):

        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, train=False)

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
        if train and self.shuffle:
            from random import shuffle
            shuffle(data_loader)

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct, db_correct, key_correct = 0, 0, 0
        total_element, total_db, total_key = 0, 0, 0
        total_strings, total_correct_strings = 0, 0
        db_acc_num, key_acc_num, all_acc_num, all_num = 0, 0, 0, 0

        for i, data in data_iter:

            # 1. forward the input and all position labels
            if self.session_loop:
                loss_pack = self.model.session_loop_forward(data)
            else:
                loss_pack = self.model(data)
            db_loss, key_loss, total_step, valid_step, db_valid_step, key_valid_step, db_correct_step, key_correct_step = \
                loss_pack['db_loss'], loss_pack['key_loss'], loss_pack[
                    'total_step'], loss_pack['valid_step'], loss_pack['db_valid_step'], loss_pack['key_valid_step'], \
                loss_pack[
                    'db_correct_step'], loss_pack['key_correct_step']

            strings_num, correct_strings_num = loss_pack['total_strings'], loss_pack['total_correct_strings']
            loss = db_loss + key_loss
            # loss = db_loss
            # 3. backward and optimization only in train
            if train:
                if not self.warmup:
                    self.optim.zero_grad()
                else:
                    self.optim_schedule.zero_grad()
                if self.bert_optim is not None: self.bert_optim.zero_grad()
                loss.backward()
                # add a grad clip
                if self.grad_clip: torch.nn.utils.clip_grad_norm_(self.params, 1)
                if not self.warmup:
                    self.optim.step()
                else:
                    self.optim_schedule.step_and_update_lr()
                if self.bert_optim is not None: self.bert_optim.step()

            avg_loss += loss.item()
            total_correct += (db_correct_step + key_correct_step)
            db_correct += db_correct_step
            key_correct += key_correct_step
            total_element += valid_step
            total_db += db_valid_step
            total_key += key_valid_step
            total_strings += strings_num
            total_correct_strings += correct_strings_num
            db_acc_num += db_correct_step/db_valid_step
            key_acc_num += key_correct_step/key_valid_step
            all_acc_num += (db_correct_step+key_correct_step)/(db_valid_step+key_valid_step)
            all_num += 1

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "Iter loss/step": loss.item() / valid_step,
                "Iter db loss/step": db_loss.item() / db_valid_step,
                "Iter key loss/step": key_loss.item() / key_valid_step,
                "Iter acc": (db_correct_step + key_correct_step) / valid_step,
                "Iter db acc": db_correct_step / db_valid_step,
                "Iter Key acc": key_correct_step / key_valid_step,
                "Iter String acc": correct_strings_num / strings_num
            }

            if train:
                self.iter += 1
                self.tensorboard_writer.add_scalar('Per Step Loss/total', post_fix['Iter loss/step'], self.iter)
                self.tensorboard_writer.add_scalar('Per Step Loss/db', post_fix['Iter db loss/step'], self.iter)
                self.tensorboard_writer.add_scalar('Per Step Loss/key', post_fix['Iter key loss/step'], self.iter)
                self.tensorboard_writer.add_scalar('Per Step Acc/total', post_fix['Iter acc'], self.iter)
                self.tensorboard_writer.add_scalar('Per Step Acc/db', post_fix['Iter db acc'], self.iter)
                self.tensorboard_writer.add_scalar('Per Step Acc/key', post_fix['Iter Key acc'], self.iter)
                self.tensorboard_writer.add_scalar('String Acc', post_fix['Iter String acc'], self.iter)

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_step_loss=" % (epoch, str_code), avg_loss / total_element,
              "step_acc=", all_acc_num / all_num,
              "db_acc=", db_acc_num / all_num, "key_acc=", key_acc_num / all_num)
        return all_acc_num / all_num, total_correct_strings / total_strings

    def save(self, epoch, step_acc, string_acc, file_path):
        """
        Saving the current  model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        # TODO：目前只实现了teacher force的acc计算过程 迭代式尚未实现
        if string_acc >= self.best_valid_acc:
            self.best_valid_acc = string_acc
            self.best_epoch = epoch
            output_path = file_path + ".ep%d.pth" % epoch
            torch.save(self.model.state_dict(), output_path)
            self.model.to(self.device)
            print("EP:%d Model Saved on:" % epoch, output_path)
            print("Current Valid String Acc is {} in {} epoch".format(string_acc, epoch))
            return output_path
        else:
            print("EP:%d Model NO Save" % epoch)
            print("Best Valid String Acc is {} in {} epoch".format(self.best_valid_acc, self.best_epoch))
            print("Current Valid String Acc is {} in {} epoch".format(string_acc, epoch))

    def load(self, epoch, file_path):
        output_path = file_path + ".ep%d.pth" % epoch
        self.model.load_state_dict(torch.load(output_path, map_location=self.device))
