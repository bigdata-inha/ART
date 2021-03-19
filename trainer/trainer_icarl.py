from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import models.model_resnet
import trainer
import trainer.trainer_warehouse as trainer_warehouse


class Trainer(trainer_warehouse.GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)
        self.args = args
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.training_idx = torch.tensor([])

        self.old_weight = []
        self.new_weight = []

        self.class_weight = np.array([])  # array for the sampling the class

        self.class_corr_dict = None

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

    def get_one_hot(self, target, num_class):
        one_hot = torch.zeros(target.shape[0], num_class).cuda()
        one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
        return one_hot

    
    def train(self, epoch, triplet):

        T = 2

        self.model.train()
        print("Epochs %d" % epoch)

        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        for data, target, traindata_idx in tqdm(self.train_data_iterator):
            data, target = data.cuda(), target.long().cuda()

            loss_CE = 0
            output = self.model(data)[:, :end]

            if tasknum == 0:
                loss_CE = self.loss(output, target)

            loss_KD = 0
            if tasknum > 0:
                '''
                data_new, target_new = data[target >= start], target[target >= start]
                output_new = output[target >= start]
                loss_CE = self.loss(output_new, target_new)

                end_KD = start
                data_old, target_old = data[target < start], target[target < start]
                output_old = output[target < start]
                score = self.model_fixed(data_old)[:,:end_KD].data

                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output_old[:, :end_KD] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')
                '''
                loss_CE = self.loss(output, target)
                end_KD = start

                score = self.model_fixed(data)[:, :end_KD].data

                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:, :end_KD] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')

            self.optimizer.zero_grad()
            (loss_KD + loss_CE).backward()
            self.optimizer.step()

        