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
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.args = args
        self.training_output = torch.tensor([])
        self.training_target = torch.tensor([])

        self.triplet_loss = torch.nn.TripletMarginLoss(margin=self.args.margin, p=2).cuda()
        # self.mr_loss = torch.nn.MarginRankingLoss(margin=self.args.margin).cuda()
        self.mr_loss = torch.nn.HingeEmbeddingLoss(margin=self.args.margin).cuda()

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

    def get_optimizer(self, optimizer):
        self.optimizer = optimizer

    def weight_align(self, new_wa=False):
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size
        lamb = start / self.args.step_size
        # weight = self.model.module.fc.weight.data
        weight = self.model.fc.weight.data

        prev = weight[:start, :]
        new = weight[start:end, :]

        self.old_weight.append(prev)
        self.new_weight.append(new)

        print(prev.shape, new.shape)
        mean_prev = torch.mean(torch.norm(prev, dim=1)).item()
        mean_new = torch.mean(torch.norm(new, dim=1)).item()
        gamma = mean_prev / mean_new
        new = new * gamma

        print(torch.mean(torch.norm(self.model.fc.weight.data[:start], dim=1)).item())
        print(torch.mean(torch.norm(self.model.fc.weight.data[start:end], dim=1)).item())

        result = torch.cat((prev, new), dim=0)
        weight[:end, :] = result

        print(torch.mean(torch.norm(self.model.fc.weight.data[:start], dim=1)).item())
        print(torch.mean(torch.norm(self.model.fc.weight.data[start:end], dim=1)).item())

    def make_class_dict(self):
        print("Making Class Corrleation Dictionary")
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        self.class_corr_dict = torch.zeros(self.args.step_size, start).cuda()
        tasknum = self.train_data_iterator.dataset.t

        if self.args.dataset == "CIFAR10":
            data_per_new = 5000
        elif self.args.dataset == "CIFAR100":
            data_per_new = 500
        elif self.args.dataset == "ImageNet100":
            data_per_new = 0

        self.model.eval()
        with torch.no_grad():
            for data, target, traindata_idx in tqdm(self.train_data_iterator):
                target = target.type(dtype=torch.long)

                data, target = data.cuda(), target.cuda()
                new_idx = torch.where(target >= start)[0]

                output = self.model(data)[:, :start]
                new_output = output[new_idx, :]
                new_prob = F.softmax(new_output, dim=1).detach()

                new_target = target[new_idx] % self.args.step_size

                for i in range(len(new_target)):
                    self.class_corr_dict[new_target[i]] += new_prob[i]

        self.class_corr_dict = self.class_corr_dict / data_per_new
        self.class_corr_dict = self.class_corr_dict.T

        old_class_corr_dict = self.class_corr_dict.cpu().numpy()
        new_class_corr_dict = self.class_corr_dict.T.cpu().numpy()

        self.old_class_distribution = np.zeros((start, self.args.step_size))
        self.new_class_distribution = np.zeros((self.args.step_size, start))

        for i in range(start):
            self.old_class_distribution[i] = old_class_corr_dict[i]

        for i in range(self.args.step_size):
            self.new_class_distribution[i] = new_class_corr_dict[i]

        print("Finished making class correlation dictionary")
        self.temp_count = np.zeros(end)
        self.i_count = 0

    def make_class_dict_CS(self, during_train=False):
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        self.class_corr_dict = torch.zeros(self.args.step_size, start).cuda()
        tasknum = self.train_data_iterator.dataset.t

        print("Defining class correlation dictionary using cosine similarity..")
        temp_anchor = copy.deepcopy(self.class_anchor)

        cos = torch.nn.CosineSimilarity(dim=0)

        if self.args.dataset == "CIFAR10":
            data_per_new = 5000
        elif self.args.dataset == "CIFAR100":
            data_per_new = 500
        elif self.args.dataset == "ImageNet100":
            data_per_new = 0

        self.class_mean_feature = torch.zeros((end, 64)).cuda()
        self.cosine_similiarity_list = []

        class_data_count = np.zeros(end)

        self.model.eval()
        with torch.no_grad():
            for data, target, traindata_idx in tqdm(self.train_data_iterator):
                target = target.type(dtype=torch.long)

                data, target = data.cuda(), target.cuda()

                _, feature_output = self.model(data, feature_return=True)

                for i in range(data.size()[0]):
                    self.class_mean_feature[target[i]] += feature_output[i].detach()
                    temp_target = target[i].cpu()
                    class_data_count[temp_target] += 1

            for j in range(end):
                self.class_mean_feature[j, :] = self.class_mean_feature[j, :] / class_data_count[j]

            self.class_mean_feature[:start] = temp_anchor[:start]

            '''
            if tasknum > 0:
                self.class_mean_feature[:start] = temp_anchor[:start]
                #self.class_mean_feature[:start] = self.class_mean_feature[:start] / int((self.args.memory_size / start))
                self.class_mean_feature[start:end] = self.class_mean_feature[start:end] / data_per_new
            else:
                self.class_mean_feature = self.class_mean_feature / data_per_new
            '''

        # calculate cosine similarity between class mean feature vector
        for i in range(self.args.step_size):
            for j in range(start):
                self.class_corr_dict[i, j] = cos(self.class_mean_feature[start + i, :], self.class_mean_feature[j, :])

        self.class_corr_dict = self.class_corr_dict.T

        old_class_corr_dict = self.class_corr_dict.cpu().numpy()
        new_class_corr_dict = self.class_corr_dict.T.cpu().numpy()

        self.old_class_distribution = np.zeros((start, self.args.step_size))
        self.new_class_distribution = np.zeros((self.args.step_size, start))

        for i in range(start):
            self.old_class_distribution[i] = old_class_corr_dict[i]

        for i in range(self.args.step_size):
            self.new_class_distribution[i] = new_class_corr_dict[i]

        print("Finished making class correlation dictionary using cosine similarity..")

    def class_corr_prob(self, batch_target):

        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        permuted_target = torch.zeros(batch_target.size()[0])

        old_choice = np.arange(start, end)  # selected by old target
        new_choice = np.arange(0, start)  # selected by new target

        for i in range(batch_target.size()[0]):
            if batch_target[i] < start:

                j = batch_target[i]
                temp = np.random.choice(old_choice,
                                        p=np.ndarray.tolist(
                                            self.old_class_distribution[j] / np.sum(self.old_class_distribution[j])))

                permuted_target[i] = temp

            else:

                j = batch_target[i] % self.args.step_size
                temp = np.random.choice(new_choice,
                                        p=np.ndarray.tolist(
                                            self.new_class_distribution[j] / np.sum(self.new_class_distribution[j])))

                permuted_target[i] = temp

        permuted_target = permuted_target.type(dtype=torch.long)
        return permuted_target

    def train(self, epoch, triplet=False):

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

            data, target = data.cuda(), target.cuda()

            old_idx = torch.where(target < start)[0]
            new_idx = torch.where(target >= start)[0]

            loss_KD = 0
            loss_CE = 0
            loss_triplet = 0
            loss_mr = 0

            if tasknum > 0 and triplet == True:
                if epoch > self.args.triplet_epoch:

                    target_a = target
                    target_b = self.class_corr_prob(target)
                    output, a_feature = self.model(data, feature_return=True)

                    p_feature = torch.zeros((data.size()[0], a_feature.size()[1])).cuda()
                    n_feature = torch.zeros((data.size()[0], a_feature.size()[1])).cuda()

                    for i in range(data.size()[0]):
                        p_feature[i] = self.class_anchor[target_a[i]]
                    p_target = torch.ones(data.size()[0], 1).cuda()
                    loss_mr = torch.mean(torch.norm((a_feature - p_feature))) * self.args.triplet_lam

                    # loss_CE = self.loss(output[:, :end], target) + loss_triplet
                    loss_CE = self.loss(output[:, :end], target)
                else:
                    output, a_feature = self.model(data, feature_return=True)

                    target_a = target
                    target_b = self.class_corr_prob(target)

                    n_feature = torch.zeros((data[new_idx, :].size()[0], a_feature.size()[1])).cuda()

                    for i in range(data[new_idx, :].size()[0]):
                        n_feature[i] = self.class_anchor[target_b[new_idx][i]]
                    n_target = torch.ones(data[new_idx, :].size()[0], 1).cuda()
                    n_target = n_target * -1

                    dist = torch.norm((a_feature[new_idx, :] - n_feature), dim=1)

                    loss_mr = self.mr_loss(dist, n_target)

                    loss_CE = self.loss(output[:, :end], target)

            else:
                output, a_feature = self.model(data, feature_return=True)
                loss_CE = self.loss(output[:, :end], target)
                loss_triplet = 0

            _, predicted = output.max(1)
            prob = F.softmax(output, dim=1).cpu()

            if tasknum > 0:
                self.model_fixed.eval()
                end_KD = start
                start_KD = end_KD - self.args.step_size

                score = self.model_fixed(data)[:, :end_KD].data

                if self.args.KD == "naive_global":
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

            if (self.args.KD == "PCKD") or (self.args.KD == "naive_global") or (self.args.KD == "naive_local"):

                if triplet:
                    # (lamb * loss_KD + (1 - lamb) * loss_CE + loss_triplet).backward()
                    (loss_KD + loss_CE + loss_triplet + loss_mr).backward()
                else:
                    (lamb * loss_KD + (1 - lamb) * loss_CE).backward()

            elif self.args.KD == "No":
                loss_CE.backward()

            self.optimizer.step()

            # weight cliping 0인걸 없애기
            weight = self.model.fc.weight.data
            weight[weight < 0] = 0
            self.model.fc.bias.data[:] = 0

        print(self.train_data_iterator.dataset.__len__())

    def make_class_Anchor(self, after_train=True):

        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        # before task change
        if tasknum > 0:
            temp_anchor = copy.deepcopy(self.class_anchor)

        if self.args.model == "resnet18":
            self.class_anchor = torch.zeros((end, 512)).cuda()
        elif self.args.model == "wideResnet":
            self.class_anchor = torch.zeros((end, 128)).cuda()
        else:
            self.class_anchor = torch.zeros((end, 64)).cuda()

        self.anchor_list = []

        if self.args.dataset == "CIFAR10":
            data_per_new = 5000
        elif self.args.dataset == "CIFAR100":
            data_per_new = 500
        elif self.args.dataset == "ImageNet100":
            data_per_new = 0

        class_data_count = np.zeros(end)

        self.model.eval()
        if after_train:  # during training because of new task
            self.model.eval()
            print("Saving Class Anchor after training..")
            with torch.no_grad():
                for data, target, traindata_idx in tqdm(self.train_data_iterator):
                    target = target.type(dtype=torch.long)

                    data, target = data.cuda(), target.cuda()

                    _, feature_output = self.model(data, feature_return=True)

                    for i in range(data.size()[0]):
                        self.class_anchor[target[i]] += feature_output[i].detach()
                        temp_target = target[i].cpu()
                        class_data_count[temp_target] += 1

                for j in range(end):
                    self.class_anchor[j, :] = self.class_anchor[j, :] / class_data_count[j]
                '''
                if tasknum > 0:
                    self.class_anchor[:start] = self.class_anchor[:start] / int((self.args.memory_size / start))
                    self.class_anchor[start:end] = self.class_anchor[start:end] / data_per_new
                else:
                    self.class_anchor = self.class_anchor / data_per_new
                '''

        else:  # during training because of new task
            print("Saving Class Anchor during training..")
            self.model.train()
            with torch.no_grad():
                for data, target, traindata_idx in tqdm(self.train_data_iterator):
                    target = target.type(dtype=torch.long)

                    data, target = data.cuda(), target.cuda()

                    _, feature_output = self.model(data, feature_return=True)

                    for i in range(data.size()[0]):
                        self.class_anchor[target[i]] += feature_output[i].detach()
                        temp_target = target[i].cpu()
                        class_data_count[temp_target] += 1

                for j in range(end):
                    self.class_anchor[j, :] = self.class_anchor[j, :] / class_data_count[j]

                self.class_anchor[:start] = temp_anchor[:start]

                '''
                if tasknum > 0:
                    print(int((self.args.memory_size / start)))
                    self.class_anchor[:start] = temp_anchor[:start]
                    # self.class_anchor[:start] = self.class_anchor[:start] / int((self.args.memory_size / start))
                    self.class_anchor[start:end] = self.class_anchor[start:end] / data_per_new
                else:
                    self.class_anchor = self.class_anchor / data_per_new
                '''

        print("Finished Saving Class Anchor..")




