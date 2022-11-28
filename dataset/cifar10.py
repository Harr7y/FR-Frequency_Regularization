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


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)


# CIFAR-10 dataset
def CIFAR10_dataset(data_dir='/home/harry/dataset/cifar10', norm=False, seed=42, val=True):
    # norm: whether normalize the data in the transform
    # val: whether split the train set into train and val set (9:1)
    if norm:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
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

    test_set = datasets.CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    if val:
        # randomly select
        np.random.seed(seed)
        split_permutation = list(np.random.permutation(50000))
        train_set = Subset(datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=True),
                           split_permutation[:45000])
        val_set = Subset(datasets.CIFAR10(data_dir, train=True, transform=test_transform, download=True),
                           split_permutation[45000:])

        return train_set, val_set, test_set

    else:
        train_set = datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        return train_set, None, test_set



# CIFAR-10 dataloader
def CIFAR10_dataloader(data_dir='/home/harry/dataset/cifar10', batch_size=128, norm=False, val=True, num_workers=0, seed=42):

    train_set, val_set, test_set = CIFAR10_dataset(data_dir, norm, seed, val)

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
    dir = '/home/harry/dataset/cifar10'
    train_loader, val_loader, test_loader = CIFAR10_dataloader(dir, 256, val=False)
    print(len(train_loader))
    print(len(train_loader.dataset))


    # [N, H, W, C]
    mean = np.mean(train_loader.dataset.data, axis=(0, 1, 2)) / 255
    std = np.std(train_loader.dataset.data, axis=(0, 1, 2)) / 255
    print(mean)
    print(std)
