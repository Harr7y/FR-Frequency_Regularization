import logging
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn


# normalization layer
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1).cuda()
        std = self.std.reshape(1, 3, 1, 1).cuda()
        return (input - mean) / std

# def normalize(X, mu, std):
#     mu = torch.tensor(mu).view(3, 1, 1).cuda()
#     std = torch.tensor(std).view(3, 1, 1).cuda()
#     return (X - mu)/std

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


# for logger
def setup_logging(log_file='log.txt', filemode='w'):
    """
    Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%m/%d %I:%M:%S %p')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# set seed
def set_seed(seed):
    np.random.seed(seed)
    # sets the seed for generating random numbers.
    torch.manual_seed(seed)
    # Sets the seed for generating random numbers for the current GPU.
    # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs.
    # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # cudnn.deterministic = True
    # cudnn.benchmark = True
    # cudnn.enabled = True


# for statistics
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# compute the accuracy of top-k
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res






