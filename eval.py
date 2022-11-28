import torch
import torch.nn
from utils import AverageMeter, accuracy
from attack.pgd import pgd_whitebox
from attack.cw import cw_whitebox
from tqdm import tqdm


# standard accuracy
def eval_clean(dataloader, model, normalize):
    top1 = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(normalize(input))
        output_clean = output_clean.float()
        # measure accuracy and record loss
        prec1 = accuracy(output_clean.data, target)[0]

        top1.update(prec1.item(), input.size(0))

    # print('eval_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


# robust accuracy
def eval_pgd(dataloader, model, normalize, epsilon, step, iters, restarts=1):
    top1 = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()

        # generate Adversarial Examples (AEs)
        X_pgd = pgd_whitebox(model, input, target, normalize=normalize,
                             epsilon=epsilon, alpha=step,
                             attack_iters=iters, restarts=restarts)

        model.eval()
        # compute output
        output_ae = model(normalize(X_pgd))
        output_ae = output_ae.float()
        # measure accuracy and record loss
        prec1 = accuracy(output_ae.data, target)[0]

        top1.update(prec1.item(), input.size(0))

    # print('eval_pgd20 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


# robust accuracy
def eval_cw(dataloader, model, normalize, epsilon, step, iters, restarts=1):
    top1 = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()

        # generate Adversarial Examples (AEs)
        X_pgd = cw_whitebox(model, input, target, normalize=normalize,
                             epsilon=epsilon, alpha=step,
                             attack_iters=iters, restarts=restarts)

        model.eval()
        # compute output
        output_ae = model(normalize(X_pgd))
        output_ae = output_ae.float()
        # measure accuracy and record loss
        prec1 = accuracy(output_ae.data, target)[0]

        top1.update(prec1.item(), input.size(0))

    # print('eval_pgd20 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg