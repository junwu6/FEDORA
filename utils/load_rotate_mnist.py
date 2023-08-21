from torchvision.datasets import FashionMNIST, MNIST
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import rotate


class RotateMNIST(Dataset):
    def __init__(self, data, targets):
        self.data, self.targets = data, targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.data)


class BaseNodes_RotateMNIST:
    def __init__(self, data_name, n_nodes, batch_size=128):
        self.data_name = data_name
        self.n_nodes = n_nodes
        self.batch_size = batch_size

        self.train_loaders, self.val_loaders, self.test_loaders = None, None, None
        self.new_train_loader, self.new_val_loader, self.new_test_loader = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        # np.random.seed(0)
        if self.data_name == "fashion-mnist":
            data_obj = FashionMNIST
            ANGLES = np.arange(0, 360.0, 360.0/self.n_nodes)
        elif self.data_name == "mnist":
            data_obj = MNIST
            ANGLES = np.arange(0, 360.0, 360.0/self.n_nodes)
        else:
            print("Unknown data")
        dataroot = "data/"

        val_size = 10000
        ridx = np.arange(60000)
        np.random.shuffle(ridx)
        train_ridx = ridx[:-val_size]
        val_ridx = ridx[-val_size:]

        test_size = 10000
        test_ridx = np.arange(test_size)
        np.random.shuffle(test_ridx)
        num_test = test_size // self.n_nodes

        train_loaders = []
        val_loaders = []
        test_loaders = []
        num_train_samples = 128
        num_val_samples = 64

        train_set = data_obj(dataroot, train=True, download=True, transform=transforms.ToTensor())
        tr_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=1000, drop_last=False)
        ori_data, ori_target = [], []
        for data, target in tr_loader:
            ori_data.append(data)
            ori_target.append(target)
        ori_data = torch.cat(ori_data, dim=0)
        ori_target = torch.cat(ori_target)

        test_set = data_obj(dataroot, train=False, download=True, transform=transforms.ToTensor())
        te_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=1000, drop_last=False)
        ori_data_test, ori_target_test = [], []
        for data, target in te_loader:
            ori_data_test.append(data)
            ori_target_test.append(target)
        ori_data_test = torch.cat(ori_data_test, dim=0)
        ori_target_test = torch.cat(ori_target_test)

        start_train = 0
        start_val = 0
        for i in range(self.n_nodes):
            # if i == self.n_nodes - 1:
            #     num_train_samples = num_train_samples * 100
            #     num_val_samples = num_val_samples*100
            end_train = start_train + num_train_samples
            data = ori_data[train_ridx[start_train: end_train]]
            client_data = rotate(data, angle=ANGLES[i])
            client_target = ori_target[train_ridx[start_train: end_train]]
            train_loader = torch.utils.data.DataLoader(dataset=RotateMNIST(client_data, client_target),
                                                       shuffle=True,
                                                       batch_size=self.batch_size,
                                                       drop_last=False)
            start_train = end_train

            end_val = start_val + num_val_samples
            val_data = ori_data[val_ridx[start_val: end_val]]
            client_val_data = rotate(val_data, angle=ANGLES[i])
            client_val_target = ori_target[val_ridx[start_val: end_val]]
            val_loader = torch.utils.data.DataLoader(dataset=RotateMNIST(client_val_data, client_val_target),
                                                     shuffle=False,
                                                     batch_size=self.batch_size,
                                                     drop_last=False)
            start_val = end_val

            test_data = ori_data_test[test_ridx[num_test*i:num_test*(i+1)]]
            client_test_data = rotate(test_data, angle=ANGLES[i])
            client_test_target = ori_target_test[test_ridx[num_test*i:num_test*(i+1)]]
            test_loader = torch.utils.data.DataLoader(dataset=RotateMNIST(client_test_data, client_test_target),
                                                      shuffle=False,
                                                      batch_size=num_test,
                                                      drop_last=False)

            if i < self.n_nodes:
                train_loaders.append(train_loader)
                val_loaders.append(val_loader)
                test_loaders.append(test_loader)
            else:
                self.new_train_loader, self.new_val_loader, self.new_test_loader = train_loader, val_loader, test_loader
        self.train_loaders, self.val_loaders, self.test_loaders = train_loaders, val_loaders, test_loaders

    def __len__(self):
        return self.n_nodes
