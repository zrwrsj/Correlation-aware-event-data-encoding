import torch
import random
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import transform_DVS
from spikingjelly_revise.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly_revise.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly_revise.datasets.n_caltech101 import NCaltech101
import spikingjelly_revise.datasets as sjds
from torch.utils.data import Dataset
import os



def cifar10_DVS(b=128, num_workers=4, data_dir = '../CIFAR10_DVS', T=16 , method='method0', segments=4):
    transform = transforms.Compose([
        transform_DVS.ToList(),
        transform_DVS.ToTensor(),
        transform_DVS.Resize(64),
    ])
    set = CIFAR10DVS(root=data_dir, data_type='frame', frames_number=T, method=method, segments=segments, split_by='number', transform=transform)
    trainset, testset = sjds.split_to_train_test_set(train_ratio=0.9, origin_dataset=set, num_classes=10, random_split=False)
    train_data_loader = DataLoader(trainset, batch_size=b, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    test_data_loader = DataLoader(testset, batch_size=b, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)
    return trainset, testset, train_data_loader, test_data_loader


def DVS_Gesture(b=128, num_workers=4, data_dir = '../DVS_Gesture', T=16, method='method2', segments=4):
    transform_train = transforms.Compose([
        transform_DVS.ToList(),
        transform_DVS.ToTensor(),
        transform_DVS.Resize(64)
    ])
    transform_test = transforms.Compose([
        transform_DVS.ToList(),
        transform_DVS.ToTensor(),
        transform_DVS.Resize(64)
    ])
    trainset = DVS128Gesture(root=data_dir, train=True, data_type='frame', frames_number=T, method=method, segments=segments, split_by='number', transform=transform_train)
    train_data_loader = DataLoader(trainset, batch_size=b, shuffle=True, num_workers=num_workers, drop_last=True,
                                   pin_memory=True)

    testset = DVS128Gesture(root=data_dir, train=False, data_type='frame', frames_number=T, method=method, segments=segments, split_by='number', transform=transform_test)
    test_data_loader = DataLoader(testset, batch_size=b, shuffle=False, num_workers=num_workers, drop_last=False,
                                  pin_memory=True)
    return trainset, testset, train_data_loader, test_data_loader


class NCaltech101_aug(Dataset):
    def __init__(self, data_path=None,
                 data_type='train', transform=True):

        data_path = data_path + "/T8_method2_s2"
        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []
        self.resize = transforms.Resize(size=(64, 64), interpolation=transforms.InterpolationMode.NEAREST)

        for i, cls in enumerate(self.clslist):
            file_list = os.listdir(os.path.join(self.filepath, cls))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for file in file_list:
                if data_type == 'train':
                    if file in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)
                else:
                    if file in test_split_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        if data_type != 'train':
            counts = np.unique(np.array(self.targets), return_counts=True)[1]
            class_weights = counts.sum() / (counts * len(counts))
            self.class_weights = torch.Tensor(class_weights)
        self.classes = range(101)
        self.transform = transform
        self.rotate = transforms.RandomRotation(degrees=15)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-15, 15))

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        data = self.resize(data)

        if self.transform:

            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-3, 3)
                off2 = random.randint(-3, 3)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                data = self.rotate(data)
            if aug == 'shear':
                data = self.shearx(data)

        return data, label

    def __len__(self):
        return self.data_num


def build_ncaltech(data_path, transform=False):
    train_dataset = NCaltech101_aug(data_path=data_path, transform=transform)
    val_dataset = NCaltech101_aug(data_path=data_path, data_type='test', transform=False)

    return train_dataset, val_dataset


def Ncaltech101(b=128, num_workers=4, data_dir = '../Ncaltech101', T=16, method='method2', segments=4):
    origin_set = NCaltech101(root=data_dir, data_type="frame", frames_number=T, method=method, segments=segments,
                             split_by="number", )

    trainset, testset = build_ncaltech(data_dir, True)
    train_data_loader = DataLoader(trainset, batch_size=b, shuffle=True, num_workers=num_workers, drop_last=True,
                                   pin_memory=True)

    test_data_loader = DataLoader(testset, batch_size=b, shuffle=False, num_workers=num_workers, drop_last=False,
                                  pin_memory=True)
    return trainset, testset, train_data_loader, test_data_loader




if __name__ == '__main__':
    trainset, _, _, _ = cifar10_DVS(T=8, method='method3', segments=4)
    num = 100
    img, label = trainset[num]
    print(img[0].shape)

