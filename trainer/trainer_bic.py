from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import models.model_resnet
import models.model_resnet18
import trainer
import trainer.trainer_warehouse as trainer_warehouse
from utils import *
from torch.distributions.distribution import *


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        self.beta = nn.Parameter(torch.Tensor(1).uniform_(0, 0))

    def forward(self, x):
        return x * self.alpha + self.beta


class Trainer(trainer_warehouse.GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)
        self.temp_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.args = args
        self.training_output = torch.tensor([])
        self.training_target = torch.tensor([])

        self.training_idx = torch.tensor([])
        self.old_weight = []
        self.new_weight = []

        self.cumulative_training_acc = torch.tensor([]).cuda()
        self.cumulative_training_target = torch.tensor([]).cuda()
        self.bias_correction_layer = BiasLayer()
        self.bias_correction_layer.cuda()
        self.bias_correction_layer_arr = []

        set_seed(args.seed)

        self.bias_optimizer = torch.optim.Adam(self.bias_correction_layer.parameters(), 0.001)

        self.class_weight = np.array([])  # array for the sampling the class

    def increment_classes(self, mode=None, bal=None, selection=None, memory_mode=None, seed=None):
        self.train_data_iterator.dataset.update_exemplar(memory_mode=memory_mode, seed=seed)

        self.train_data_iterator.dataset.task_change()
        self.test_data_iterator.dataset.task_change()
        self.bias_correction_layer_arr.append(self.bias_correction_layer)

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f" % (self.current_lr,
                                                                          self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def update_bias_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] * 2 == epoch:
                for param_group in self.bias_optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f" % (self.current_lr,
                                                                          self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def setup_training(self, lr):
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f" % lr)
            param_group['lr'] = lr
            self.current_lr = lr
        lr = lr
        # self.bias_correction_layer = BiasLayer().cuda()
        # self.bias_optimizer = torch.optim.Adam(self.bias_correction_layer.parameters(), lr)
        self.bias_optimizer = torch.optim.SGD(self.bias_correction_layer.parameters(), self.args.lr,
                                              momentum=self.args.momentum)

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()

        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def get_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, epoch, triplet=False, bft=False):

        T = 2
        self.model.train()
        print("Epochs %d" % epoch)

        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        lamb = start / end

        self.training_target = torch.tensor([]).cuda()
        self.training_output = torch.tensor([]).cuda()
        self.training_idx = torch.tensor([]).cuda()

        for data, target, traindata_idx in tqdm(self.train_data_iterator):
            target = target.type(dtype=torch.long)

            old_idx = torch.where(target < start)[0]
            new_idx = torch.where(target >= start)[0]

            data, target = data.cuda(), target.cuda()
            data.requires_grad = True

            loss_KD = 0

            output, a_feature = self.model(data, feature_return=True)
            loss_CE = self.loss(output[:, :end], target)

            if tasknum > 0:
                end_KD = start
                start_KD = end_KD - self.args.step_size

                layer = self.bias_correction_layer_arr[-1]  # last bias correction layer

                score = self.model_fixed(data)[:, :end_KD].data
                score = torch.cat([score[:, :start_KD], layer(score[:, start_KD:end_KD])], dim=1)

                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:, :end_KD] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')

            self.optimizer.zero_grad()
            if (self.args.KD == "naive_local") or (self.args.KD == "naive_global"):
                (lamb * loss_KD + (1 - lamb) * loss_CE).backward()

            elif self.args.KD == "No":
                loss_CE.backward()

            self.optimizer.step()
        print(self.train_data_iterator.dataset.__len__())

    def train_bias_correction(self, bias_iterator):

        self.model.eval()
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        for data, target, traindata_idx in tqdm(bias_iterator):
            data, target = data.cuda(), target.long().cuda()

            output = self.model(data)[:, :end]

            # bias correction
            output_new = self.bias_correction_layer(output[:, start:end])
            output = torch.cat((output[:, :start], output_new), dim=1)

            loss_CE = self.loss(output, target)

            self.bias_optimizer.zero_grad()
            (loss_CE).backward()
            self.bias_optimizer.step()

        # self.train_data_iterator.dataset.mode = 'train'