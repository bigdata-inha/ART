import argparse
from trainer.train import *
from models.model_resnet import *
import myData.iDataset
import myData.iDataLoader
from utils import *
from sklearn.utils import shuffle
import trainer.trainer_warehouse
import trainer.evaluator
from arguments import *


#parser.add_argument("")
args = get_args()

#seed
seed = args.seed
set_seed(seed)

dataset = myData.iDataset.CIFAR10()

shuffle_idx = shuffle(np.arange(dataset.classes), random_state = seed)

#shuffle_idx = np.genfromtxt('C:/Users/Hongjun/Desktop/Cifar100_SuperClass_labelnum.csv',delimiter=',',encoding="UTF-8", skip_header=0, dtype = np.int32)
#shuffle_idx[0] = 20

print(shuffle_idx)

tasknum = (dataset.classes - args.start_classes) // args.step_size + 1

myNet = resnet32(num_classes=dataset.classes, tasknum=tasknum).cuda()

if args.dataset == 'CIFAR100':
    loader = None
else:
    loader = dataset.loader

train_dataset_loader = myData.iDataLoader.IncrementalLoader(dataset.train_data,
                                                            dataset.train_labels,
                                                            dataset.classes,
                                                            args.step_size,
                                                            args.memory_size,
                                                            'train',
                                                            transform=dataset.train_transform,
                                                            loader=loader,
                                                            shuffle_idx=shuffle_idx,
                                                            base_classes=args.start_classes,
                                                            approach= "wa",
                                                            )

evaluate_dataset_loader = myData.iDataLoader.IncrementalLoader(dataset.train_data,
                                                            dataset.train_labels,
                                                            dataset.classes,
                                                            args.step_size,
                                                            args.memory_size,
                                                            'train',
                                                            transform=dataset.train_transform,
                                                            loader=loader,
                                                            shuffle_idx=shuffle_idx,
                                                            base_classes=args.start_classes,
                                                            approach= "wa",
                                                            )

test_dataset_loader = myData.iDataLoader.IncrementalLoader(dataset.test_data,
                                                            dataset.test_labels,
                                                            dataset.classes,
                                                            args.step_size,
                                                            args.memory_size,
                                                            'test',
                                                            transform=dataset.test_transform,
                                                            loader=loader,
                                                            shuffle_idx=shuffle_idx,
                                                            base_classes=args.start_classes,
                                                            approach= "wa",
                                                            )

result_dataset_loaders = myData.iDataLoader.make_ResultLoaders(dataset.test_data,
                                                         dataset.test_labels,
                                                         dataset.classes,
                                                         args.step_size,
                                                         transform=dataset.test_transform,
                                                         loader=loader,
                                                         shuffle_idx = shuffle_idx,
                                                         base_classes = args.start_classes
                                                        )


train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True, drop_last=False)
evaluator_iterator = torch.utils.data.DataLoader(evaluate_dataset_loader, batch_size=args.batch_size, shuffle=True, drop_last=False)
test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=100, shuffle=False)

optimizer = optim.SGD(myNet.parameters(), args.lr, momentum=0.9,
                        weight_decay=5e-4, nesterov=True)

myTrainer = trainer.trainer_warehouse.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myNet, args, optimizer)

testType = "trainedClassifier"

myEvaluator = trainer.evaluator.EvaluatorFactory.get_evaluator(testType, classes=dataset.classes)

train_start = 0
train_end = args.start_classes
test_start = 0
test_end = args.start_classes
total_epochs = args.nepochs
schedule = np.array(args.schedule)

results = {}

for head in ['all', 'prev_new', 'task', 'cheat']:
    results[head] = {}
    results[head]['correct'] = []
    results[head]['correct_5'] = []
    results[head]['stat'] = []

results['task_soft_1'] = np.zeros((tasknum, tasknum))
results['task_soft_5'] = np.zeros((tasknum, tasknum))


print(tasknum)

correct_list = []
stat_list = []

task_confidence_list = []
get_confidence = False

task_error = []

import matplotlib.pyplot as plt

w = 10
h = 10
fig = plt.figure(figsize=(10, 10))
columns = 10
rows = 10
# temp_sortHard = var_grad_classRank[0][:26]


#img = myTrainer.train_data_iterator.dataset.data[(3 * 500 - 1)]
#plt.imshow(img)

for i in range(1, 10):
    img = myTrainer.train_data_iterator.dataset.data[(i*5000 - 1)]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)


plt.show()