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

        set_seed(args.seed)
        self.class_weight = np.array([])  # array for the sampling the class

    def increment_classes(self, mode=None, bal=None, selection=None, memory_mode=None, seed=None):
        self.train_data_iterator.dataset.update_exemplar(memory_mode=memory_mode, seed=seed)

        self.train_data_iterator.dataset.task_change()
        self.test_data_iterator.dataset.task_change()

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
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

                score = self.model_fixed(data).data

                if bft == True:
                    score = self.model_fixed(data).data

                    soft_target = F.softmax(score[:, end_KD:end] / T, dim=1)
                    output_log = F.log_softmax(output[:, end_KD:end] / T, dim=1)
                    loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')

                elif self.args.KD == "naive_global":

                    soft_target = F.softmax(score / T, dim=1)
                    output_log = F.log_softmax(output[:, :end_KD] / T, dim=1)
                    loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')

                elif self.args.KD == "naive_local":
                    local_KD = torch.zeros(tasknum).cuda()
                    for t in range(tasknum):
                        local_start_KD = (t) * self.args.step_size
                        local_end_KD = (t + 1) * self.args.step_size

                        soft_target = F.softmax(score[:, local_start_KD:local_end_KD] / T, dim=1)
                        output_log = F.log_softmax(output[:, local_start_KD:local_end_KD] / T, dim=1)
                        local_KD[t] = F.kl_div(output_log, soft_target, reduction="batchmean")

                    loss_KD = local_KD.sum()

            self.optimizer.zero_grad()

            if (self.args.KD == "naive_local") or (self.args.KD == "naive_global"):
                (0.5 * loss_KD + loss_CE).backward()

            elif self.args.KD == "No":
                loss_CE.backward()

            self.optimizer.step()
        print(self.train_data_iterator.dataset.__len__())

    def balance_fine_tune(self):
        t = self.train_data_iterator.dataset.t
        self.update_frozen_model()
        self.setup_training(self.args.bft_lr)

        self.train_data_iterator.dataset.update_bft_buffer()
        self.train_data_iterator.dataset.mode = 'b-ft'

        schedule = np.array(self.args.schedule)
        #         bftepoch = int(self.args.nepochs*3/4)

        for epoch in range(30):
            self.update_lr(epoch, schedule)
            self.train(epoch, bft=True)

        self.train_data_iterator.dataset.mode = 'train'