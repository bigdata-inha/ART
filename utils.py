from utils import *
import os
import copy
from tqdm import tqdm_notebook
import math
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch
import numpy as np
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def set_seed(seed):
    """
    Set a random seed for numpy and PyTorch.
    """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)




def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_confusion_matrix(target, output, n_label, epochs) :
    stacked = torch.stack(
        (target, output), dim =1
    )
    cmt = torch.zeros(n_label,n_label, dtype = torch.int64)

    for p in stacked :
        tl,pl = p.tolist()
        cmt[tl,pl] = cmt[tl,pl] +1

    cmt = cmt/epochs

    classes = np.arange(0,n_label)

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cmt, classes)

def get_entropy(batch_prob) :
    entropy = torch.sum((batch_prob * torch.log(batch_prob)), dim=1)
    entropy = -entropy

    return entropy

def get_model(model):
    return copy.deepcopy(model.state_dict())
