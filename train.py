import time
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from attack.pgd import pgd_whitebox
from utils import AverageMeter, accuracy


# natural training
def train_epoch(model, dataloader, criterion, optimizer, normalize):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()
        # compute output
        output_clean = model(normalize(input))
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        output = output_clean.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    # print('train_accuracy {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg


# adversarial training
def train_epoch_adv(args, model, dataloader, criterion, optimizer, normalize):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()
        # generate Adversarial Examples (AEs)
        adv_example = pgd_whitebox(model, input, target, normalize=normalize,
                                   epsilon=args.train_epsilon, alpha=args.train_alpha,
                                   attack_iters=args.train_iters, restarts=args.restarts)
        model.train()
        # compute output
        adv_example = normalize(adv_example)

        optimizer.zero_grad()
        output_ae = model(adv_example)
        loss = criterion(output_ae, target)

        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        output_ae = output_ae.float()
        loss = loss.float()
        prec1 = accuracy(output_ae.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    # print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


# AT + FR
def train_epoch_at_FR(args, model, dataloader, criterion, optimizer, normalize):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()
        # generate Adversarial Examples (AEs)
        adv_example = pgd_whitebox(model, input, target, normalize=normalize,
                                   epsilon=args.train_epsilon, alpha=args.train_alpha,
                                   attack_iters=args.train_iters, restarts=args.restarts)
        model.train()
        # compute output
        optimizer.zero_grad()
        # ae: adversarial example
        output_ae = model(normalize(adv_example))
        loss = criterion(output_ae, target)

        # ***************************************************** #
        # Frequency Regularization
        output_clean = model(normalize(input))
        adv_fft = torch.rfft(output_ae, signal_ndim=2, normalized=False, onesided=False)
        clean_fft = torch.rfft(output_clean, signal_ndim=2, normalized=False, onesided=False)

        fre_loss = torch.nn.L1Loss()(adv_fft, clean_fft)
        loss += args.fre_rate * fre_loss
        # ***************************************************** #

        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        output_ae = output_ae.float()
        loss = loss.float()
        prec1 = accuracy(output_ae.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    # print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg

