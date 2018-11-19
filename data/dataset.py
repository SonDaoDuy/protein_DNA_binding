import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train, is_for_val):
        if dataset_name == 'dna':
            from data.dataset_dna import DNADataset
            dataset = DNADataset(opt, is_for_train, is_for_val)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, is_for_train, is_for_val):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._opt = opt
        self._is_for_train = is_for_train
        self._is_for_val = is_for_val
        self._create_transform()

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform
