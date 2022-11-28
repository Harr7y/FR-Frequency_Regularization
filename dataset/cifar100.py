import torch
import numpy as np
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, Subset


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


# CIFAR-100 dataset
def CIFAR100_dataset(data_dir='/home/harry/dataset/cifar100', norm=False, seed=42, val=True):
    if norm:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    if val:
        np.random.seed(seed)
        split_permutation = list(np.random.permutation(50000))
        train_set = Subset(datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=True),
                           split_permutation[:45000])
        val_set = Subset(datasets.CIFAR100(data_dir, train=True, transform=test_transform, download=True),
                           split_permutation[45000:])
        test_set = datasets.CIFAR100(data_dir, train=False, transform=test_transform, download=True)

        return train_set, val_set, test_set

    else:
        train_set = datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR100(data_dir, train=False, transform=test_transform, download=True)
        return train_set, None, test_set



# CIFAR-10 dataloader
def CIFAR100_dataloader(data_dir='/home/harry/dataset/cifar100', batch_size=128, norm=False, val=True, num_workers=0, seed=42):

    train_set, val_set, test_set = CIFAR100_dataset(data_dir, norm, seed, val)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                               shuffle=True, pin_memory=True,
                                               num_workers=num_workers,)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,
                                              shuffle=False, pin_memory=True,
                                              num_workers=num_workers,)

    if val:
        val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size,
                                              shuffle=False, pin_memory=True,
                                              num_workers=num_workers,)
    else:
        assert val_set == None
        print("No validation set used.")
        val_loader = None

    return train_loader, val_loader, test_loader




if __name__ == '__main__':
    dir = '/home/harry/dataset/cifar100'
    train_loader, val_loader, test_loader = CIFAR100_dataloader(dir, 256, val=False)
    print(len(train_loader))
    print(len(train_loader.dataset))

    # [N, H, W, C]
    mean = np.mean(train_loader.dataset.data, axis=(0, 1, 2)) / 255
    std = np.std(train_loader.dataset.data, axis=(0, 1, 2)) / 255
    print(mean)
    print(std)
