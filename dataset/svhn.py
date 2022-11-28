import torch
import numpy as np
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

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


svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.1980, 0.2010, 0.1970)



# SVHN dataset
def SVHN_dataset(data_dir='/home/harry/dataset/svhn', norm=False, seed=42, val=True):
    if norm:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
    else:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    if val:
        np.random.seed(seed)
        # total number: 73257
        split_permutation = list(np.random.permutation(65932))
        train_set = Subset(datasets.SVHN(data_dir, split='train', transform=train_transform, download=True),
                           split_permutation[:65932])
        val_set = Subset(datasets.SVHN(data_dir, split='train', transform=test_transform, download=True),
                           split_permutation[65932:])
        test_set = datasets.SVHN(data_dir, split='test', transform=test_transform, download=True)

        return train_set, val_set, test_set

    else:
        train_set = datasets.SVHN(data_dir, split='train', transform=train_transform, download=True)
        test_set = datasets.SVHN(data_dir, split='test', transform=test_transform, download=True)
        return train_set, None, test_set



# SVHN dataloader
def SVHN_dataloader(data_dir='/home/harry/dataset/svhn', batch_size=128, norm=False, num_workers=0, seed=42, val=True):

    train_set, val_set, test_set = SVHN_dataset(data_dir, norm, seed, val)

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
    dir = '/home/harry/dataset/svhn'
    train_loader, val_loader, test_loader = SVHN_dataloader(dir, 256, val=False)
    print(len(train_loader))
    print(len(train_loader.dataset))

    # [N, C, H, W]
    mean = np.mean(train_loader.dataset.data, axis=(0, 2, 3)) / 255
    std = np.std(train_loader.dataset.data, axis=(0, 2, 3)) / 255
    print(mean)
    print(std)
