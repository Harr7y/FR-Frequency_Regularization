import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ResNet import ResNet18
from model.Wide_ResNet import WideResNet

from utils import *
from eval import *
from dataset.cifar10 import CIFAR10_dataloader
from autoattack import AutoAttack
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


def normalize_PGDAT(X):
    return (X - mu) / std


def normalize_TRADES(X):
    return X


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--model', default='resnet', choices=['resnet', 'wrn'],
                        help='directory of model for saving checkpoint')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--trial', default=0, type=int, help='experiment index')
    parser.add_argument('--norm_type', default='PGDAT', choices=['TRADES', 'PGDAT'], type=str)
    parser.add_argument('--ckpt_path', help='The checkpoint path')
    return parser.parse_args()


def main():
    args = get_args()

    set_seed(args.seed)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler()
        ])

    logger.info(args)

    data_dir = '/home/harry/dataset/cifar10'
    _, _, test_loader = CIFAR10_dataloader(data_dir, args.batch_size,val=False)

    model_dir = args.model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # args.ckpt_path = '/home/harry/nnet/AAAI_workshop/ckpt/resnet/0/best_model_0.pt'
    best_state_dict = torch.load(args.ckpt_path)
    if 'state_dict' in best_state_dict:
        best_state_dict= best_state_dict['state_dict']

    if args.norm_type == 'TRADES':
        normalize = normalize_TRADES
    elif args.norm_type == 'PGDAT':
        normalize = normalize_PGDAT


    # Evaluation
    if args.model =='resnet':
        model_test = ResNet18().cuda()
    elif args.model == 'wrn':
        model_test = WideResNet().cuda()

    model_test.load_state_dict(best_state_dict, strict=True)

    model_test.float()
    model_test.eval()

    ### Evaluate clean acc ###
    test_acc = eval_clean(test_loader, model_test, normalize)
    print('Clean acc: ', test_acc)

    ## Evaluate PGD (CE loss) acc ###
    pgd_acc_CE = eval_pgd(test_loader, model_test, normalize, 8, 2, 20, 1)
    print('PGD-20 (1 restarts, step 2, CE loss) acc: ', pgd_acc_CE)


    pgd_acc_CE = eval_pgd(test_loader, model_test, normalize, 8, 2, 50, 1)
    print('PGD-50 (1 restarts, step 2, CE loss) acc: ', pgd_acc_CE)

    ### Evaluate CW (CW loss) acc ###
    pgd_acc_CW = eval_cw(test_loader, model_test, normalize, 8, 2, 30, 1)
    print('CW-30 (1 restarts, step 2, CW loss) acc: ', pgd_acc_CW)

    ### Evaluate AutoAttack ###
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    class normalize_model():
        def __init__(self, model):
            self.model_test = model

        def __call__(self, x):
            return self.model_test(normalize(x))

    new_model = normalize_model(model_test)
    epsilon = 8 / 255.
    adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard')
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)


if __name__ == "__main__":
    main()