import copy
import logging
import time
import math

import numpy as np
import torch
import torch.utils.data as td
from sklearn.utils import shuffle
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms.functional as trnF
from torch.nn import functional as F
from utils import *


class IncrementalLoader(td.Dataset):
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, transform=None, loader=None, shuffle_idx=None,
                 model=None,
                 base_classes=50, approach='wa'):
        if shuffle_idx is not None:
            # label shuffle
            print("Label shuffled")
            labels = shuffle_idx[labels]
        set_seed(1)
        sort_index = np.argsort(labels)
        self.data = data[sort_index]

        labels = np.array(labels)
        self.labels = labels[sort_index]

        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.loader = loader
        self.total_classes = classes

        self.labels_arr = np.unique(self.labels)  # for sampling method

        self.step_size = step_size
        self.base_classes = base_classes
        self.t = 0

        self.mem_sz = mem_sz
        self.validation_buffer_size = int(mem_sz / 10) * 1
        self.mode = mode

        self.start = 0
        self.end = base_classes

        self.start_idx = 0
        self.end_idx = np.argmax(self.labelsNormal > (self.end - 1))  # end data index

        if self.end == classes:
            self.end_idx = len(labels) - 1

        self.tr_idx = range(self.end_idx)
        self.len = len(self.tr_idx)
        self.current_len = self.len

        self.approach = approach
        self.memory_buffer = []
        self.exemplar = []
        self.validation_buffer = []
        self.start_point = []
        self.end_point = []

        for i in range(classes):
            self.start_point.append(np.argmin(self.labelsNormal < i))
            self.end_point.append(np.argmax(self.labelsNormal > (i)))
            self.memory_buffer.append([])
        self.end_point[-1] = len(labels)

        # variables for sampling strategy
        self.count_forgetting = np.zeros(len(self.data))
        self.count_forgetting_list = []

        self.prev_forgetting = np.zeros(len(self.data))
        self.new_forgetting = np.zeros(len(self.data))

        self.class_index_list = []
        self.class_forgetting_count = []
        self.cls_sb_hard_idx = []

        self.grad_list = []
        self.temp_input_grad = np.zeros(len(self.data))
        self.class_grad_list = []
        self.class_grad_checkList = []

        self.KD_prob_diff = np.zeros(len(self.data))
        self.class_kd_diff_list = []

    def task_change(self):
        self.t += 1

        self.start = self.end
        self.end += self.step_size

        print('dataset start, end: ', self.start, self.end)

        self.start_idx = np.argmin(self.labelsNormal < self.start)  # start data index
        self.end_idx = np.argmax(self.labelsNormal > (self.end - 1))  # end data index
        if self.end_idx == 0:
            self.end_idx = self.labels.shape[0]

        self.tr_idx = range(self.start_idx, self.end_idx)

        # validation set for bic
        if self.approach == 'bic' and self.start < self.total_classes and self.mode != 'test':
            val_per_class = (self.validation_buffer_size // 2) // self.step_size
            self.tr_idx = []
            for i in range(self.step_size):
                end = self.end_point[self.start + i]
                start = self.start_point[self.start + i]
                self.validation_buffer += range(end - val_per_class, end)
                self.tr_idx += range(start, end - val_per_class)

            print('exemplar, validation: ', len(self.exemplar), len(self.validation_buffer))

            arr = []
            for idx in self.validation_buffer:
                arr.append(self.labelsNormal[idx])

        self.len = len(self.tr_idx)
        self.current_len = self.len

        if self.approach == 'ft' or self.approach == 'icarl' or self.approach == 'bic' or self.approach == 'wa' or self.approach == 'eeil' or self.approach == "CTL" or self.approach == "vanilla" or self.approach == "joint":
            print(self.approach)
            self.len += len(self.exemplar)

    def update_bft_buffer(self):
        self.bft_buffer = copy.deepcopy(self.memory_buffer)

        min_len = 1e8
        for arr in self.bft_buffer:
            min_len = max(min_len, len(arr))

        buffer_per_class = min_len

        buffer_per_class = math.ceil(self.mem_sz / self.start)
        if buffer_per_class > 500:
            buffer_per_class = 500

        for i in range(self.start, self.end):
            start_idx = self.start_point[i]
            end_idx = self.end_point[i]
            idx = range(start_idx, start_idx + buffer_per_class)

            self.bft_buffer[i] += list(idx)

        for arr in self.bft_buffer:
            if len(arr) > buffer_per_class:
                arr.pop()

        self.bft_exemplar = []

        for arr in self.bft_buffer:
            self.bft_exemplar += arr

    def update_exemplar(self, memory_mode=None, seed=None):

        buffer_per_class = math.ceil(self.mem_sz / self.end)
        if buffer_per_class > 5000:
            buffer_per_class = 5000
        # first, add new exemples

        for i in range(self.start, self.end):
            start_idx = self.start_point[i]
            self.memory_buffer[i] += range(start_idx, start_idx + buffer_per_class)

        # second, throw away the previous samples
        if buffer_per_class > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class:
                    del self.memory_buffer[i][buffer_per_class:]

        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!

        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz
        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c].pop()

        self.exemplar = []
        for arr in self.memory_buffer:
            self.exemplar += arr

        # validation set for bic
        if self.approach == 'bic':
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def update_exemplar_classAdaptively(self, mode=None, bal=None, selection=None, memory_mode=None, method=None,
                                        throw_away=None):
        print("Exemplar Choosing by adaptively DM+KD")
        print("start : {}, end : {}".format(self.start, self.end))

        change_rank = True

        if memory_mode == "fixed":
            buffer_per_class = 20
        else:
            buffer_per_class = math.ceil(self.mem_sz / self.end)

        print("buffer_per_class :", buffer_per_class)

        print("Arrange the KD information")
        kd_diff_List = self.KD_prob_diff.copy()

        print(np.where(kd_diff_List > 0)[0].shape)

        print("Arrange the gradient information")
        gradHist = np.stack(self.grad_list, axis=0)

        weight = np.arange(1, 11)
        weight = 1 / weight

        weight_avg_grad = np.average(gradHist, weights=weight, axis=0)  # weight avg

        var_grad = np.var(gradHist, axis=0)  # varaince

        lamb = 0.5
        newMeasure = (lamb * self.normalize(weight_avg_grad)) + (
                (1 - lamb) * self.normalize(var_grad))  # to be revised later

        check_gradList = np.sum(gradHist, axis=0)

        if method == "weight_avg":
            gradList = weight_avg_grad
        elif method == "var":
            gradList = var_grad
        elif method == "metric":
            gradList = newMeasure

        # class_min = int(buffer_per_class / 5)

        # shared_memory = (buffer_per_class - class_min) * self.end
        # for the unbalanced but reasonable data set
        # class_weight = class_min + np.array(class_weight * shared_memory)

        # buffer_per_class = class_weight.astype(np.int)

        print(np.where(check_gradList > 0)[0].shape)

        temp = 0

        exemplarRatio = buffer_per_class / 500

        print("exemplar Ratio : ", exemplarRatio)

        temp = 0
        self.class_kd_diff_list = []
        self.class_index_list = []
        self.cls_sb_hard_idx = []

        self.class_grad_list = []
        self.class_grad_checkList = []
        # throw away previous exemplar by highKD diff
        for i in (self.labels_arr[0:self.end]):

            self.class_index_list.append(np.where(self.labelsNormal == i)[0])

            self.class_kd_diff_list.append(kd_diff_List[self.class_index_list[i]])

            self.class_grad_list.append(gradList[self.class_index_list[i]])

            self.class_grad_checkList.append(
                check_gradList[self.class_index_list[i]])  # sum of gradient to check whether it

            check_KD_list = np.where(self.class_kd_diff_list[i] > 0)

            kd_argsort = np.argsort(self.class_kd_diff_list[i][check_KD_list])[::-1]  # descending order

            selection_check_idx = check_KD_list[0][kd_argsort]

            selection_check = self.class_kd_diff_list[i][selection_check_idx].mean()

            if self.class_kd_diff_list[i][selection_check_idx][0] < self.class_kd_diff_list[i][selection_check_idx][1]:
                print("Something is wrong please Check")

            if selection_check > 20:  # hyperparameter
                selection_check_flag = "diversity"
            else:
                selection_check_flag = "highKD"

            ##############################################################################################################################
            # Choose by high KD
            if selection_check_flag == "diversity":
                print("Choose the exemplar by diveristy maximization")
                rm_class_forget = np.where(self.class_grad_checkList[temp] > 0)

                temp_argsort = np.argsort(self.class_grad_list[temp][rm_class_forget])

                temp_argsort = rm_class_forget[0][temp_argsort]

                temp_index = []

                for j in range(buffer_per_class):
                    '''
                    if exemplarRatio < 0.1 :
                        exemplar_end = len(temp_argsort) * 0.5                       
                    elif (exemplarRatio < 0.5) and (exemplarRatio >= 0.1) :
                        exemplar_end = len(temp_argsort) * 0.75
                    else : 
                        exemplar_end = len(temp_argsort)
                    '''
                    exemplar_end = len(temp_argsort)

                    num_index = exemplar_end / buffer_per_class

                    a = int(j * num_index)

                    temp_index.append(a)

                temp_index = np.stack(temp_index)

                temp_argsort = temp_argsort[temp_index]
                temp_class_index = self.class_index_list[i][temp_argsort]

            ##########################################################################################################################################
            # Choose by Diversity Maximization
            else:
                print("Choose the exemplar by highKD")

                temp_argsort = selection_check_idx.copy()

                # temp_argsort = np.argsort(rm_class_forget)  # ascending order // select the hard one

                # temp_argsort = np.argsort(rm_class_forget)[::-1]  # descending order // select the easy one

                temp_value = int(500 - int(buffer_per_class))  # select Hard

                temp_argsort = temp_argsort[:buffer_per_class]  # Easy
                bad_index = temp_argsort[:(buffer_per_class - 10)]

                temp_class_index = self.class_index_list[i][temp_argsort]

                print("high diff mean :", self.KD_prob_diff[temp_class_index].mean())
                print("more high diff mean : ", self.KD_prob_diff[self.class_index_list[i][bad_index]].mean())

            self.cls_sb_hard_idx.append(temp_class_index)

            temp += 1

        for i in range(0, self.end):
            start_idx = self.start_point[i]
            self.memory_buffer[i] = self.cls_sb_hard_idx[i].copy()

        '''
        if change_rank == False:
            # second, throw away the previous samples
            if buffer_per_class > 0:
                for i in range(self.start):
                    if len(self.memory_buffer[i]) > buffer_per_class:
                        self.memory_buffer[i] = self.memory_buffer[i][:buffer_per_class]
        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!
        '''

        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz

        total_mem = 0
        for i in range(self.end):
            total_mem = total_mem + buffer_per_class

        print(total_mem)
        print(remain)

        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c] = self.memory_buffer[c][:-1]
                # self.memory_buffer[c] = self.memory_buffer[c].pop()

        self.exemplar = []

        for arr in self.memory_buffer[:self.end]:
            print(arr)
            arr = arr.tolist()
            self.exemplar += arr

        print(self.exemplar.__len__())
        # self.exemplar = np.concatenate(self.exemplar)
        # print(self.exemplar)

        # validation set for bic
        if 'bic' in self.approach:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []

            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def normalize(self, input):
        ma = np.max(input)
        mi = np.min(input)
        output = (input - mi) / (ma - mi)
        return output

    def __len__(self):
        if self.mode == 'train':
            return self.len
        elif self.mode == 'bias':
            return len(self.validation_buffer)
        elif self.mode == 'b-ft':
            return len(self.bft_exemplar)
        else:
            return self.end_idx

    def __getitem__(self, index):
        #         time.sleep(0.1)
        if self.mode == 'train':
            if index >= self.current_len:  # for bic, ft, icarl, il2m
                index = self.exemplar[index - self.current_len]
            else:
                index = self.tr_idx[index]
        elif self.mode == 'bias':  # for bic bias correction
            index = self.validation_buffer[index]
        elif self.mode == 'b-ft':
            index = self.bft_exemplar[index]
        img = self.data[index]

        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index], index


class ResultLoader(td.Dataset):
    def __init__(self, data, labels, transform=None, loader=None):

        self.data = data
        self.labels = labels
        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.loader = loader
        self.transformLabels()

    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((self.labels.size, self.labels.max() + 1))
        b[np.arange(self.labels.size), self.labels] = 1
        self.labels = b

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        #         time.sleep(0.1)
        img = self.data[index]
        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index], index


def make_ResultLoaders(data, labels, classes, step_size, transform=None, loader=None, shuffle_idx=None,
                       base_classes=50):
    if shuffle_idx is not None:
        labels = shuffle_idx[labels]
    sort_index = np.argsort(labels)
    data = data[sort_index]
    labels = np.array(labels)
    labels = labels[sort_index]

    start = 0
    end = base_classes

    loaders = []

    while (end <= classes):

        start_idx = np.argmin(labels < start)  # start data index
        end_idx = np.argmax(labels > (end - 1))  # end data index
        if end_idx == 0:
            end_idx = data.shape[0]

        loaders.append(
            ResultLoader(data[start_idx:end_idx], labels[start_idx:end_idx], transform=transform, loader=loader))

        start = end
        end += step_size

    return loaders


def iterator(dataset_loader, batch_size, shuffle=False, drop_last=False):
    kwargs = {'num_workers': 0, 'pin_memory': False}
    return torch.utils.data.DataLoader(dataset_loader, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                       **kwargs)