import argparse
# import deepspeed

def get_args():
    ########basic argument##################
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.1, type=float, \
                        help="optimizer learning rate")

    parser.add_argument("--memory_size", default=200, type=int, \
                        help="exemplar set memory size")

    parser.add_argument("--nepochs", default=50, type=int, \
                        help="number of Epochs")

    parser.add_argument('--schedule', type=int, nargs='+', default=[30,40],
                        help='learning rate decay epoch')

    parser.add_argument("--batch_size", default=128, type=int,
                        help="batch size for training")

    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

    parser.add_argument('--bft_lr', default = 0.01, type=float,
                        help='learning rate decay epoch')

    parser.add_argument('--bft_schedule', type=int, nargs='+', default=[20,30],
                        help='learning rate decay epoch')

    parser.add_argument('--bft_gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

    parser.add_argument("--step_size", default=2, type=int, \
                        help="number class for one continual learning steps")

    parser.add_argument("--start_classes", default=2, type=int,
                        help="number of class in one step")

    parser.add_argument("--dataset", default="CIFAR10", choices=['CIFAR10', 'CIFAR100', "Imagenet"], type=str,
                        help='myData name')

    parser.add_argument("--seed", default=1, type=float, help='seed for reproducibility')

    parser.add_argument("--KD", default="naive_global", choices = ['naive_local', 'naive_global', "No"], type=str, help ="which distillation")

    parser.add_argument("--trainer", default="vanilla", choices=["vanilla", "wa", 'icarl', 'eeil' "CTL", "bic"],
                        type=str, help="which CIL method use?")

    parser.add_argument('--triplet', default=False, type=bool, help ='cctriplet or none')

    parser.add_argument('--margin', default=0.5, type=float,
                        help='distance margin for tripelt loss')

    parser.add_argument('--triplet_lam', default=1, type=float,
                        help = 'ratio of triplet loss')

    parser.add_argument('--dict_type', default = "softmax", choices=["softmax", 'cosine'], help="which metric to define dictionary")

    parser.add_argument('--dict_update', default=True, type=bool,
                        help="whether update dictionary or not")

    parser.add_argument('--new_WA', default="False", type=bool,
                        help="wheter use new WA")

    parser.add_argument('--distance', default ="Eucledian", choices=["Eucledian", "cosine"], help="which metric to use for the triplet loss")
    parser.add_argument('--model', default= "resnet32", choices=["resnet32", "resnet18", "wideResnet"], help="model define")
    parser.add_argument('--triplet_epoch', default = 10, type=float, help="when to start train using triplet loss")
    parser.add_argument('--anchor_update_epoch', default = 10 , type=float, help="anchor update period")
    args=parser.parse_args()

    return args