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

# parser.add_argument("")
args = get_args()

# seed
seed = args.seed
set_seed(seed)

dataset = myData.iDataset.CIFAR100()

shuffle_idx = shuffle(np.arange(dataset.classes), random_state=seed)

myNet = resnet32().cuda()

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
                                                            approach="wa",
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
                                                               approach="wa",
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
                                                           approach="wa",
                                                           )

result_dataset_loaders = myData.iDataLoader.make_ResultLoaders(dataset.test_data,
                                                               dataset.test_labels,
                                                               dataset.classes,
                                                               args.step_size,
                                                               transform=dataset.test_transform,
                                                               loader=loader,
                                                               shuffle_idx=shuffle_idx,
                                                               base_classes=args.start_classes
                                                               )

train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True)
evaluator_iterator = torch.utils.data.DataLoader(evaluate_dataset_loader, batch_size=args.batch_size, shuffle=True)
test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=100, shuffle=False)

optimizer = optim.SGD(myNet.parameters(), args.lr, momentum=0.9,
                      weight_decay=5e-4, nesterov=True)

myTrainer = trainer.trainer_warehouse.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myNet, args,
                                                                 optimizer)

testType = "generativeClassifier"
myEvaluator = trainer.evaluator.EvaluatorFactory.get_evaluator(testType, classes=dataset.classes)

train_start = 0
train_end = args.start_classes
test_start = 0
test_end = args.start_classes
total_epochs = args.nepochs
schedule = np.array(args.schedule)

tasknum = (dataset.classes - args.start_classes) // args.step_size + 1

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
print(args.KD)
#################Get Into CL!###############################
for t in range(tasknum) :
    get_confidence = False

    log_name = '{}_{}_{}_base_{}_step_{}_batch_{}_epoch_{}'.format(
        args.trainer,
        args.dataset,
        args.seed,
        args.start_classes,
        args.step_size,
        args.batch_size,
        args.nepochs,
    )



    correct = {}   #record for correct
    stat = {}      #record for statistics e.g. ep, enn ..
    confidence = []#record for confidence based on task

    lr = args.lr

    myTrainer.update_frozen_model()
    myTrainer.setup_training(lr)

    flag = 0

    print("SEED:", args.seed, "MEMORY_BUDGET:", args.memory_size, "tasknum:", t)


    for epoch in range(args.nepochs) :
        myTrainer.update_lr(epoch, args.schedule)
        myTrainer.train(epoch)

        if epoch % 5 == 4 :
            myEvaluator.update_moment(myTrainer.model, evaluator_iterator, args.step_size, t)
            if t == 0 :
                train_1 = myEvaluator.evaluate(myTrainer.model, train_iterator, test_start, test_end,
                                           mode='train', step_size=args.step_size, tasknum=tasknum)

                test_1 = myEvaluator.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                           mode='test', step_size=args.step_size, tasknum=tasknum)

                print("*********CURRENT EPOCH********** : %d" % epoch)
                print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
                print("Test Classifier top-1 (Softmax): ", test_1)

            else :
                train_1 = myEvaluator.evaluate(myTrainer.model, evaluator_iterator, 0, train_end, get_confidence=get_confidence, tasknum=tasknum)
                correct, stat = myEvaluator.evaluate(myTrainer.model, test_iterator,
                                                      test_start, test_end,
                                                      mode='test', step_size=args.step_size, tasknum=tasknum)

                print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
                print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
                print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
                print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
                print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
                print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])


    if t > 0 and (args.trainer == "icarl"):
        myEvaluator.update_moment(myTrainer.model, evaluator_iterator, args.step_size, t)
        correct, stat = myEvaluator.evaluate(myTrainer.model, test_iterator,
                                              test_start, test_end,
                                              mode='test', step_size=args.step_size, tasknum=tasknum)
        print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
        print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
        print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
        print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
        print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

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

    for i in range(t + 1):
        dataset_loader = result_dataset_loaders[i]
        iterator = torch.utils.data.DataLoader(dataset_loader,
                                               batch_size=args.batch_size)

        if 'bic' in args.trainer:
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = myEvaluator.evaluate(myTrainer.model,
                                                                                               iterator, start, end,
                                                                                               myTrainer.bias_correction_layer, tasknum=tasknum)
        else:
            results['task_soft_1'][t][i] = myEvaluator.evaluate(myTrainer.model, iterator, start, end, mode="test", tasknum=tasknum)

        start = end
        end += args.step_size

        #with open('./checkpoint/result_data/' + log_name, "wb") as f:
        #    f.write(results)
        #    f.close()

        #f = open('.checkpoint/result_data/{}_task_{}_output.txt'.format(log_name, t), "w")
        #f.write(str(results))
        #f.close()

    correct_list.append(correct)
    stat_list.append(stat)
    #task_confidence_list.append(confidence)

    torch.save(myNet.state_dict(), './checkpoint/comparasion/' + 'CIFAR10_my_{}_FixM_{}_{}_{}.pt'.format(args.trainer, args.KD, tasknum, t))

    #myTrainer.increment_classes(mode="grad", selection="SeqEasy", bal="bal", memory_mode = "Fbuffer", method = "var")
    myTrainer.increment_classes(mode="Bal", bal="None", memory_mode = None)
    #myTrainer.increment_classes(mode="forgetting_bal", selection="SeqEasy", bal="bal", memory_mode="Fbuffer", method="var")
    #myTrainer.increment_classes(mode="kd", bal="None", selection="highDiff", memory_mode = "Fbuffer")

    #myTrainer.increment_classes(mode="adaptive_KD+DM", bal="None", selection=None, memory_mode = "Fbuffer", method="metric")

    evaluate_dataset_loader.update_exemplar()
    evaluate_dataset_loader.task_change()


    #for bic
    #bias_dataset_loader.update_exemplar()
    #bias_dataset_loader.task_change()

    train_end = train_end + args.step_size
    test_end = test_end + args.step_size

# Check Confidence List
task_confidence_matrix = []
for i in range(tasknum):
    task_mean = 0
    temp_list = []
    for j in range(task_confidence_list[i].__len__()):
        task_mean += round(task_confidence_list[i][j].item(), 4)
        if j % args.step_size == (args.step_size - 1):
            temp_value = round(task_mean / args.step_size, 4)
            temp_list.append(temp_value)
            task_mean = 0

    task_confidence_matrix.append(temp_list)

current_state = "naive, random, 5step, Balance, stage2~4, all_data_distill"

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