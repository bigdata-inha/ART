import argparse
from trainer.train import *
from models.model_resnet import *
from models.model_resnet18 import *
import myData.iDataset
import myData.iDataLoader
from utils import *
from sklearn.utils import shuffle
import trainer.trainer_warehouse
import trainer.evaluator
from arguments import *
from myData.data_warehouse import *
from models.model_Wresnet import *
import torch.optim as optim

#parser.add_argument("")
args = get_args()

#seed
seed = args.seed
set_seed(seed)

#set gpu
GPU_NUM = args.GPU_NUM  # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(device)
print("current cuda : ", torch.cuda.current_device())

if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM) / 1024 ** 3, 1), 'GB')

data = DatasetWH
dataset = data.get_dataset(args.dataset)

shuffle_idx = shuffle(np.arange(dataset.classes), random_state=seed)

tasknum = (dataset.classes - args.start_classes) // args.step_size + 1

#######################################################################dataset, dataloader, model decalare
if args.dataset == 'CIFAR100' or args.dataset == 'CIFAR10':
    loader = None
    myNet = resnet32(num_classes=dataset.classes, tasknum=tasknum).cuda()
else:
    loader = dataset.loader
    myNet = wideresnet(depth=16, num_classes=200, widen_factor=2, dropRate=0.3).cuda()


train_dataset_loader = myData.iDataLoader.IncrementalLoader(dataset.train_data,
                                                            dataset.train_labels,
                                                            dataset.classes,
                                                            args.step_size,
                                                            args.memory_size,
                                                            'train',
                                                            args = args,
                                                            transform=dataset.train_transform,
                                                            loader=loader,
                                                            shuffle_idx=shuffle_idx,
                                                            base_classes=args.start_classes,
                                                            approach= args.trainer,
                                                            )

evaluate_dataset_loader = myData.iDataLoader.IncrementalLoader(dataset.train_data,
                                                            dataset.train_labels,
                                                            dataset.classes,
                                                            args.step_size,
                                                            args.memory_size,
                                                            'train',
                                                            args=args,
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
                                                           args = args,
                                                            transform=dataset.test_transform,
                                                            loader=loader,
                                                            shuffle_idx=shuffle_idx,
                                                            base_classes=args.start_classes,
                                                            approach= args.trainer,
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

#only for the BiC
bias_dataset_loader = myData.iDataLoader.IncrementalLoader(dataset.train_data,
                                                           dataset.train_labels,
                                                           dataset.classes,
                                                           args.step_size,
                                                           args.memory_size,
                                                           'bias',
                                                           transform=dataset.train_transform,
                                                           loader=loader,
                                                           shuffle_idx=shuffle_idx,
                                                           base_classes=args.start_classes,
                                                           approach=args.trainer
                                                           )

train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True,
                                             drop_last=True)
evaluator_iterator = torch.utils.data.DataLoader(evaluate_dataset_loader, batch_size=args.batch_size, shuffle=True)
test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=50, shuffle=False)
#######################################################################################################################

####################################################################################Set optimizer, trainer, evaluator

optimizer = optim.SGD(myNet.parameters(), args.lr, momentum=0.9,
                        weight_decay=5e-4, nesterov=True)

if args.trainer == "icarl" :
    test_type = "generativeClassifier"
else :
    testType = "trainedClassifier"

myTrainer = trainer.trainer_warehouse.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myNet, args, optimizer)
myEvaluator = trainer.evaluator.EvaluatorFactory.get_evaluator(testType, classes=dataset.classes)
bic_Evaluator = trainer.evaluator.EvaluatorFactory.get_evaluator("bic", classes=dataset.classes)
#######################################################################################################################

####################################################################################etc informaation


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


correct_list = []
stat_list = []

task_confidence_list = []
get_confidence = False

task_error = []

#################Get Into Incremental Learning!###############################
print("datset : ", args.dataset, "| trainer : ", args.trainer, "| kdloss : ", args.KD, " | CCtriplet : ", args.CCtriplet)
for t in range(tasknum):
    get_confidence = False

    correct = {}  # record for correct
    stat = {}  # record for statistics e.g. ep, enn ..

    lr = args.lr

    myTrainer.update_frozen_model()
    myTrainer.setup_training(lr)

    if t > 0: #make class correlation matrix
        if args.dict_type == "softmax":
            myTrainer.make_class_dict()

    print("SEED:", args.seed, "MEMORY_BUDGET:", args.memory_size, "tasknum:", t)

    for epoch in range(args.nepochs):
        myTrainer.update_lr(epoch, args.schedule)
        myTrainer.train(epoch, triplet=args.CCtriplet)

        if epoch % 5 == 4:
            if args.trainer == "icarl":
                myEvaluator.update_moment(myTrainer.model, evaluator_iterator, args.step_size, t)
            if t == 0:
                get_confidence = False
                train_1 = myEvaluator.evaluate(myTrainer.model, evaluator_iterator, 0, train_end,
                                                   get_confidence=get_confidence, tasknum=tasknum)
                test_1 = myEvaluator.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                              mode='test', step_size=args.step_size, tasknum=tasknum)

                print("*********CURRENT EPOCH********** : %d" % epoch)
                print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
                print("Test Classifier top-1 (Softmax): ", test_1)

            else:
                if epoch == args.nepochs - 1 & get_confidence == True:
                    get_confidence = False
                    train_1, confidence = myEvaluator.evaluate(myTrainer.model, evaluator_iterator, 0, train_end,
                                                               get_confidence=get_confidence, tasknum=tasknum)
                else:
                    train_1 = myEvaluator.evaluate(myTrainer.model, evaluator_iterator, 0, train_end,
                                                   get_confidence=get_confidence, tasknum=tasknum)

                correct, stat = myEvaluator.evaluate(myTrainer.model, test_iterator,
                                                     test_start, test_end,
                                                     mode='test', step_size=args.step_size, tasknum=tasknum)

                print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
                print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
                print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
                print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
                print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
                print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

        if (epoch == args.anchor_update_epoch - 1) and t > 0 and args.triplet == True :
            myTrainer.make_class_Anchor(after_train=False)  # save class anchor for next step

            if args.dict_update == True:
                if args.dict_type == "softmax":
                    myTrainer.make_class_dict()
                else:
                    myTrainer.make_class_dict_CS()

    if t > 0 and (args.trainer == 'wa' or args.trainer == "CLT"): #weight align for bias correction
        myTrainer.weight_align(new_wa=args.new_WA)
        correct, stat = myEvaluator.evaluate(myTrainer.model, test_iterator,
                                             test_start, test_end,
                                             mode='test', step_size=args.step_size, tasknum=tasknum)
        print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
        print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
        print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
        print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
        print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

    if t > 0 and (args.trainer == 'eeil'): #balanced finutning fot EEIL
        myTrainer.balance_fine_tune()
        correct, stat = myEvaluator.evaluate(myTrainer.model, test_iterator,
                                             test_start, test_end,
                                             mode='test', step_size=args.step_size, tasknum=tasknum)
        print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
        print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
        print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
        print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
        print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])
        print("Test Classifier top-1 (Softmax, ti_correct): %0.2f" % correct['task_id_correct'])

    if args.trainer == 'bic' and t > 0:

            best_acc = 0

            bias_iterator = myData.iDataLoader.iterator(bias_dataset_loader, batch_size=args.batch_size, shuffle=True)

            print(myTrainer.bias_correction_layer.alpha)
            print(myTrainer.bias_correction_layer.beta)

            for e in range(args.nepochs * 2):
                myTrainer.train_bias_correction(bias_iterator)
                myTrainer.update_bias_lr(e, schedule)

                if e % 5 == (4):
                    correct, stat = bic_Evaluator.evaluate(myTrainer.model, test_iterator,
                                                           test_start, test_end, myTrainer.bias_correction_layer,
                                                           mode='test', step_size=args.step_size)
                    print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
                    print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
                    print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
                    print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
                    print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

            correct, stat = bic_Evaluator.evaluate(myTrainer.model, test_iterator,
                                                   test_start, test_end, myTrainer.bias_correction_layer,
                                                   mode='test', step_size=args.step_size)
            print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
            print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
            print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
            print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
            print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

    if args.triplet == True :
        if t > 0 :
            myTrainer.make_class_Anchor(after_train=False)  # save class anchor for next step
        else :
            myTrainer.make_class_Anchor()

    if t > 0:
        correct, stat = myEvaluator.evaluate(myTrainer.model, test_iterator,
                                             test_start, test_end,
                                             mode='test', step_size=args.step_size, tasknum=tasknum)
        for head in ['all', 'pre', 'new', 'intra_pre', 'intra_new']:
            results['all']['correct'].append(correct[head])
        results['all']['stat'].append(stat['all'])

    else:
        test_1 = myEvaluator.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                      mode='test', step_size=args.step_size, tasknum=tasknum)
        print("Test Classifier top-1 (Softmax): ", test_1)
        for head in ['all']:
            results[head]['correct'].append(test_1)

    start = 0
    end = args.start_classes

    correct_list.append(correct)
    stat_list.append(stat)

    if args.CCtriplet == True:
        torch.save(myNet.state_dict(),
                   './checkpoint/comparasion/' + 'base_{}_{}_tri{}_trilam{}_newWA{}_{}_{}_{}.pt'.format(args.trainer,
                                                                                                     args.dataset,
                                                                                                     args.CCtriplet,
                                                                                                     args.triplet_lam,
                                                                                                     args.new_WA,
                                                                                                     tasknum, t, args.model))
    else:
        torch.save(myNet.state_dict(),
                   './checkpoint/comparasion/' + 'base_{}_{}_tri{}_newWA{}_{}_{}_{}.pt'.format(args.trainer, args.dataset,
                                                                                            args.CCtriplet, args.new_WA,
                                                                                            tasknum, t, args.model))

    myTrainer.increment_classes(mode="Bal", bal="None", memory_mode=None)

    evaluate_dataset_loader.update_exemplar()
    evaluate_dataset_loader.task_change()

    # for bic
    bias_dataset_loader.update_exemplar()
    bias_dataset_loader.task_change()

    train_end = train_end + args.step_size
    test_end = test_end + args.step_size

print(args)
print()
print("print acc")
for i in range(tasknum):
    if i == 0:
        print(test_1)
    else:
        print(correct_list[i]["all"])

print()
print("print all")
for i in range(tasknum):
    if i == 0:
        print(test_1)
    else:
        print(correct_list[i]["intra_pre"], " ", correct_list[i]["intra_new"], " ", correct_list[i]["pre"], " ",
              correct_list[i]["new"], " ", correct_list[i]["task_id_correct"])

'''
for i in range(args.step_size):
    trainer.beforeTrain()
    accuracy = trainer.train(i)
    trainer.afterTrain()
    
'''

'''
 if args.step_size < 5  :
     if t > 0:
         train_1 = myEvaluator.evaluate_top1(myTrainer.model, test_iterator, 0, train_end)
         print("*********CURRENT EPOCH********** : %d" % epoch)
         print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
         #print("Train Classifier top-5 (Softmax): %0.2f" % train_5)

         correct, stat = myEvaluator.evaluate_top1(myTrainer.model, test_iterator,
                                                      test_start, test_end,
                                                      mode='test', step_size=args.step_size)

         print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])

         print("Test Classifier top-1 (Softmax, prev_new): %0.2f" % correct['prev_new'])


         for head in ['all', 'prev_new', 'task']:
             results[head]['correct'].append(correct[head])
             #results[head]['correct_5'].append(correct_5[head])
             results[head]['stat'].append(stat[head])

     else:
         ###################### 폐기처분 대상 ######################
         train_1 = myEvaluator.evaluate_top1(myTrainer.model, train_iterator, 0, train_end)
         print("*********CURRENT EPOCH********** : %d" % epoch)
         print("Train Classifier top-1 (Softmax): %0.2f" % train_1)

         test_1 = myEvaluator.evaluate_top1(myTrainer.model, test_iterator, test_start, test_end,
                                                mode='test', step_size=args.step_size)
         print("Test Classifier top-1 (Softmax): %0.2f" % test_1)


         for head in ['all', 'prev_new', 'task', 'cheat']:
             results[head]['correct'].append(test_1)

 else :
     if t > 0:
         train_1, train_5 = myEvaluator.evaluate(myTrainer.model, test_iterator, 0, train_end)
         print("*********CURRENT EPOCH********** : %d" % epoch)
         print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
         print("Train Classifier top-5 (Softmax): %0.2f" % train_5)

         correct, correct_5, stat = myEvaluator.evaluate(myTrainer.model, test_iterator,
                                                         test_start, test_end,
                                                         mode='test', step_size=args.step_size)

         print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
         print("Test Classifier top-5 (Softmax, all): %0.2f" % correct_5['all'])
         print("Test Classifier top-1 (Softmax, prev_new): %0.2f" % correct['prev_new'])
         print("Test Classifier top-5 (Softmax, prev_new): %0.2f" % correct_5['prev_new'])

         for head in ['all', 'prev_new', 'task']:
             results[head]['correct'].append(correct[head])
             results[head]['correct_5'].append(correct_5[head])
             results[head]['stat'].append(stat[head])

     else:
         ###################### 폐기처분 대상 ######################
         train_1, train_5 = myEvaluator.evaluate(myTrainer.model, train_iterator, 0, train_end)
         print("*********CURRENT EPOCH********** : %d" % epoch)
         print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
         print("Train Classifier top-5 (Softmax): %0.2f" % train_5)

         test_1, test_5 = myEvaluator.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                               mode='test', step_size=args.step_size)
         print("Test Classifier top-1 (Softmax): %0.2f" % test_1)
         print("Test Classifier top-5 (Softmax): %0.2f" % test_5)

         for head in ['all', 'prev_new', 'task', 'cheat']:
             results[head]['correct'].append(test_1)
             results[head]['correct_5'].append(test_5)
 '''