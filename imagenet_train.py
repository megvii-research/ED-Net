import argparse
import os
import random
import shutil
import time
import warnings

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from utils import time_file_str

from ed_resnet import *
from ed_resnext import *
from ed_se_resnet import *
from ed_se_resnext import *

model_dict = {
    'ED50':resnet50_ed(),
    'EDX50':resnext50_ed(),
    'EDSE50':se_resnet50_ed(),
    'EDSEX50':se_resnext50_ed(),
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--test_data_path', default='')
parser.add_argument('--train_data_path', default='')
parser.add_argument('--save_dir', type=str, default='./logs', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N')
parser.add_argument('--epochs', default=100, type=int, metavar='N')
parser.add_argument('--schedule', default=30, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,metavar='LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')

parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--eval', default=0, type=int, help='eval=1 to eval, eval=0 to train')

best_prec1 = 0
best_prec5 = 0

def main():
    global args, best_prec1, best_prec5
    args = parser.parse_args()
    args.prefix = time_file_str()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch, args.prefix)), 'w')
    log_top1err = open(os.path.join(args.save_dir, '{}.{}.top1err-log'.format(args.arch, args.prefix)), 'w')
    log_top5err = open(os.path.join(args.save_dir, '{}.{}.top5err-log'.format(args.arch, args.prefix)), 'w')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    # create model
    model = model_dict[args.arch]
    print_log("[model] '{}'".format(args.arch), log)
    print_log("{}".format(model), log)
    print_log("[args parameter] : {}".format(args), log)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(size=224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]
    )

    test_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]
    )

    def image_loader_PIL(path):#for test
        return Image.open(path).convert('RGB')

    class DSet(Dataset):
        def __init__(self, image_list='', transform=None, loader=image_loader_PIL, data_path=''):
            file = open(image_list, 'r')
            imgs = []
            for string in file:
                string = string.strip('\n')
                string = string.rstrip()
                sample = string.split()
                imgs.append((sample[0], int(sample[1])))
            self.imgs = imgs
            self.transform = transform
            self.loader = loader
            self.data_path = data_path

        def __getitem__(self, index):
            image_name, label = self.imgs[index]
            image_name = self.data_path + image_name
            img = self.loader(image_name)
            img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    train_data = DSet(image_list='your_train_data_list.txt', transform=train_transform, loader=image_loader_PIL, data_path=args.train_data_path)
    val_data = DSet(image_list='your_test_data_list.txt', transform=test_transform, loader=image_loader_PIL, data_path=args.test_data_path)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.eval == 1:
        validate(val_loader, model, criterion, log, log_top1err, log_top5err)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, log, log_top1err, log_top5err)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = max(prec5, best_prec5)
        print('[==epoch==]', epoch, '[==best_top1==] ', 100-best_prec1, '[==best_top5==] ', 100-best_prec5)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5),log=log)


def validate(val_loader, model, criterion, log, logtop1, logtop5):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5),log)

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        print_log('[Top1-Acc] {top1.avg:.3f} [Top5-Acc] {top5.avg:.3f} [Top1-Error] {error1:.3f} [Top5-Error] {error5:.3f}]'.format
                  (top1=top1, top5=top5, error1=100 - top1.avg, error5=100 - top5.avg), log)

        print_log('{error1:.3f}'.format(error1=100 - top1.avg), logtop1)
        print_log('{error5:.3f}'.format(error5=100 - top5.avg), logtop5)

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, args):
    torch.save(state, args.arch + '.pth.tar')
    if is_best:
        shutil.copyfile(args.arch + '.pth.tar', args.arch + 'best.pth.tar')

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every x epochs"""
    lr = args.lr * (0.1 ** (epoch // args.schedule))
    print('learning rate:',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
