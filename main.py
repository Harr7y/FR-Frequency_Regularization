import os
import time
import torch
import logging
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model.Wide_ResNet import WideResNet
from model.ResNet import ResNet18

from dataset.svhn import SVHN_dataloader
from dataset.cifar10 import CIFAR10_dataloader
from dataset.cifar100 import CIFAR100_dataloader

from loss.LabelSmooth import LabelSmoothingLoss
from eval import eval_clean, eval_pgd
from utils import setup_logging, set_seed, Normalize
from train import train_epoch, train_epoch_adv, train_epoch_at_FR
from WeightAverage import moving_average, bn_update


parser = argparse.ArgumentParser(description='PyTorch CIFAR AT+FR Defense')
# training hyperparameters
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr_drop', type=str, default='75, 90', metavar='LR',
                    help='learning rate drop epoch')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--l2', type=bool, default=False, help='L2 norm loss')

# PGD hyperparameters
parser.add_argument('--train_epsilon', default=8, type=int, help='perturbation')
parser.add_argument('--train_iters', default=10, type=int, help='perturb number of steps')
parser.add_argument('--train_alpha', default=2, type=int, help='perturb step size')
parser.add_argument('--restarts', default=1, type=int, help='restart attack number')
parser.add_argument('--test_epsilon', default=8, type=int, help='perturbation')
parser.add_argument('--test_iters', default=20, type=int, help='perturb number of steps')
parser.add_argument('--test_alpha', default=2, type=int, help='perturb step size')

# loss
parser.add_argument('--loss', default='ce', choices=['ce', 'ols','ls'], help='different types of loss function')
# label smoothing
parser.add_argument('--labelsmooth', default=0.2, type=float, help='label smooth value')

# frequency regularization
parser.add_argument('--fre_loss', action='store_true', help='logits frequency pair loss')
parser.add_argument('--fre_rate', default=0.1, type=float, help='rate of frequency loss')
parser.add_argument('--fre_start_epoch', default=75, type=int, help='FR starts epoch number (default: 75)')

# weight average
parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=75, metavar='N', help='SWA starts epoch number (default: 75)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N', help='SWA model collection frequency/cycle length in epochs (default: 1)')

# checkpoint
parser.add_argument('--resume', action='store_true', help='load state dict from the checkpoint')
parser.add_argument('--ckpt_path', type=str, default='./ckpt/', help='checkpoint path')

# others
parser.add_argument('--model', default='resnet', choices=['resnet', 'wrn', 'preresnet', 'resnet_ff'],
                    help='directory of model for saving checkpoint')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'tin'], help='dataset name')
parser.add_argument('--dataset_path', default='/home/harry/dataset/', help='dataset path')
parser.add_argument('--tb_dir', type=str, default='./tb/', help='the tensorboard log')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--trial', type=int, default=0, help='the experimental index')

parser.add_argument('--eval_test', action='store_true', help='whether check the accuracy of test set')
parser.add_argument('--periodic_save', action='store_true', help='checkpoint frequently saved flag (default: off)')
parser.add_argument('--save_frequency', type=int, default=10, help='checkpoint save frequency')
parser.add_argument('--norm_type', default='TRADES', choices=['TRADES', 'PGDAT'], type=str)

args = parser.parse_args()

# settings
# checkpoint
ckpt_dir = args.ckpt_path + args.model
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_trial_path = os.path.join(ckpt_dir, str(args.trial))
if not os.path.exists(ckpt_trial_path):
    os.mkdir(ckpt_trial_path)
# logger
log_dir = './log/' + args.model
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logfile = os.path.join(log_dir, str(args.trial) + '.txt')
setup_logging(logfile)
logging.info(args)

# tensorboard
tb_dir = args.tb_dir + args.model
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
tb_path = os.path.join(tb_dir, str(args.trial))
writer = SummaryWriter(tb_path)


set_seed(args.seed)
lr_drop = list(map(int, args.lr_drop.split(',')))
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

## load dataset

# setup data loader and norm layer
if args.dataset == 'svhn':
    train_loader, val_loader, test_loader = SVHN_dataloader(args.dataset_path + 'svhn', batch_size=args.batch_size)
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)
    num_classes = 10
elif args.dataset == 'cifar10':
    train_loader, val_loader, test_loader = CIFAR10_dataloader(args.dataset_path + 'cifar10', batch_size=args.batch_size)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_loader, val_loader, test_loader = CIFAR100_dataloader(args.dataset_path + 'cifar100', batch_size=args.batch_size)
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    num_classes = 100

if args.norm_type == 'TRADES':
    normalize = Normalize((0., 0., 0.), (1.0, 1.0, 1.0))
elif args.norm_type == 'PGDAT':
    normalize = Normalize(mean, std)


## load model
if args.model =='resnet':
    model = ResNet18(num_classes=num_classes)
    if args.swa:
        swa_model = ResNet18(num_classes=num_classes)

elif args.model == 'wrn':
    model = WideResNet(num_classes=num_classes)
    if args.swa:
        swa_model = WideResNet(num_classes=num_classes)

# DDP unfinished yet
# model = nn.DataParallel(model)
model = model.cuda()
if args.swa:
    # swa_model = nn.DataParallel(swa_model)
    swa_model = swa_model.cuda()


# optimizer
if args.l2:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if 'bn' not in name and 'bias' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    params = [{'params': decay, 'weight_decay':0.0005},
              {'params': no_decay, 'weight_decay': 0}]
else:
    params = model.parameters()
optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


## resume
epochs = 0
# load checkpoint
if args.resume:
    state = torch.load(args.ckpt_path)
    epochs = state['epoch'] + 1
    ckpt = state["state_dict"]
    # optimizer.load_state_dict(state["optimizer"])
    optimizer.load_state_dict(state["optimizer"].state_dict())

    newckpt = {}
    for k,v in ckpt.items():
        if "module." in k:
            newckpt[k.replace("module.", "")] = v
        else:
            newckpt[k] = v
    del ckpt
    model.load_state_dict(newckpt, strict=True)


# surrogate loss function
if args.loss == 'ce':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'ls':
    criterion = LabelSmoothingLoss(smoothing=args.labelsmooth)

# learning rate
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr_drop1, lr_drop2 = lr_drop
    lr = args.lr
    if epoch >= lr_drop2:
        lr = args.lr * 0.01
    elif epoch >= lr_drop1:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# train and record
def main():
    logging.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    if args.swa:
        swa_n = 0
    best_pgd_acc = 0
    best_swa_pgd_acc = 0

    for epoch in range(epochs, args.epochs):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()

        # adversarial training
        if args.fre_loss and epoch >= args.fre_start_epoch:
            train_acc1, train_loss = train_epoch_at_FR(args, model, train_loader, criterion, optimizer,
                                                     normalize=normalize)
        else:
            train_acc1, train_loss = train_epoch_adv(args, model, train_loader, criterion, optimizer, normalize=normalize)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                     epoch, time.time() - start_time, lr, train_loss, train_acc1)

        writer.add_scalar('Train/accuracy', train_acc1, epoch)
        writer.add_scalar('Train/loss', train_loss, epoch)
        print('================================================================')

        # Compute the accuracy on the val set and record
        eval_clean_acc1 = eval_clean(val_loader, model, normalize)
        eval_pgd_acc1 = eval_pgd(val_loader, model, normalize, args.test_epsilon, args.test_alpha,
                                args.test_iters, args.restarts)
        logging.info('Eval accuracy: \t %.4f, Eval robustness: \t %.4f', eval_clean_acc1, eval_pgd_acc1)
        writer.add_scalar('Eval/accuracy', eval_clean_acc1, epoch)
        writer.add_scalar('Eval/robustness', eval_pgd_acc1, epoch)

        if eval_pgd_acc1 > best_pgd_acc:
            best_pgd_acc = eval_pgd_acc1
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()},
                       os.path.join(ckpt_trial_path, 'best_model_' + str(args.trial) +'.pt'))
        print('using time:', time.time() - start_time)

        # periodic save
        if args.periodic_save and (epoch) % args.save_frequency == 0:
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()},
                       os.path.join(ckpt_trial_path, str(epoch) + '_epoch_' + str(args.trial) +'.pt'))


        # weight average
        if args.swa and epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0:
            # SWA
            # print("SWA_N:", swa_n)
            moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            bn_update(train_loader, swa_model, normalize)
            swa_model.eval()

            eval_swa_clean_acc1 = eval_clean(val_loader, swa_model, normalize)
            eval_swa_pgd_acc1 = eval_pgd(val_loader, swa_model, normalize, args.test_epsilon, args.test_alpha,
                                args.test_iters, args.restarts)
            logging.info('SWA: Eval accuracy: \t %.4f, Eval robustness: \t %.4f', eval_swa_clean_acc1, eval_swa_pgd_acc1)

            writer.add_scalar('Eval/SWA_SA',  eval_swa_clean_acc1, epoch)
            writer.add_scalar('Eval/SWA_RA', eval_swa_pgd_acc1, epoch)

            if eval_swa_pgd_acc1 > best_swa_pgd_acc:
                best_swa_pgd_acc = eval_swa_pgd_acc1
                torch.save({'state_dict': swa_model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()},
                           os.path.join(ckpt_trial_path, 'swa_best_model_' + str(args.trial) + '.pt'))

        elif args.swa: # store the accuracy of standard model to complete the curve.
            writer.add_scalar('Eval/SWA_SA', eval_clean_acc1, epoch)
            writer.add_scalar('Eval/SWA_RA', eval_pgd_acc1, epoch)


        # check the accuracy of the test set during the training
        if args.eval_test:
            test_clean_acc1 = eval_clean(test_loader, swa_model, normalize)
            test_pgd_acc1 = eval_pgd(test_loader, swa_model, normalize, args.test_epsilon, args.test_alpha,
                                args.test_iters, args.restarts)
            logging.info('Test accuracy: \t %.4f, Test robustness: \t %.4f', test_clean_acc1,
                         test_pgd_acc1)

            writer.add_scalar('Test/SA', test_clean_acc1, epoch)
            writer.add_scalar('Test/RA', test_pgd_acc1, epoch)

            if args.swa and epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0:
                test_swa_clean_acc1 = eval_clean(test_loader, swa_model, normalize)
                test_swa_pgd_acc1 = eval_pgd(test_loader, swa_model, normalize, args.test_epsilon, args.test_alpha,
                                args.test_iters, args.restarts)
                logging.info('SWA: Test accuracy: \t %.4f, Test robustness: \t %.4f', test_swa_clean_acc1, test_swa_pgd_acc1)

                writer.add_scalar('Test/SWA_SA',  test_swa_clean_acc1, epoch)
                writer.add_scalar('Test/SWA_RA', test_swa_pgd_acc1, epoch)
            elif args.swa:  # sto+re the accuracy of standard model to complete the curve.
                writer.add_scalar('Test/SWA_SA', test_clean_acc1, epoch)
                writer.add_scalar('Test/SWA_RA', test_pgd_acc1, epoch)


    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()