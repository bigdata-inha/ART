import myData.iDataset as data



class DatasetWH:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name):
        if name == 'CIFAR100':
            return data.CIFAR100()
        elif name == 'CIFAR10':
            return data.CIFAR10()
        elif name == "TinyImagenet":
            return data.TinyImagenet()
