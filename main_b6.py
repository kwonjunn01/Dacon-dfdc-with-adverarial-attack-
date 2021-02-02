import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models
import cv2
import torchvision.transforms as transforms

from tqdm import tqdm
import easydict
from dataset import DFDCDatatset
from efficientnet_pytorch import EfficientNet
import os.path as osp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import *


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))





args = easydict.EasyDict({
    "arch": 'resnet18',
    "root": "/home/diml/ddrive/dataset/deepfake_1st",
    "train_list": "train_list_1st.txt",
    "test_list": "test_list_1st.txt",
    "epochs": 200,
    "batch_size": 2,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    
    "print_freq": 100,
    "eval_freq": 10,
    
    "workers": 4,
    #"resume":"10fgsm_b7_checkpoint.pth.tar",
    "resume": False,
    "pretrained": False,
    "evaluate": False,
    "start_epoch": 0,
    "gpu": 0,
})

best_acc1 = 0
# for fgsm image generation
def fgsm_attack(model, loss, images, labels, eps) :
    

    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels).cuda()
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images

# check for present learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    global args, best_acc1
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = EfficientNet.from_pretrained('efficientnet-b6')
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # Using NAG
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov =True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = DFDCDatatset(args.root,
                              args.train_list,
                              transforms.Compose([
                                  transforms.CenterCrop(500),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))

    test_dataset = DFDCDatatset(args.root,
                            args.test_list,
                            transforms.Compose([
                                transforms.CenterCrop(500),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    

    if args.evaluate:
        part_te = torch.utils.data.random_split(test_dataset, [4000, len(test_dataset)-4000])[0]
        test_loader = torch.utils.data.DataLoader(
        part_te, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False)
        validate(test_loader, model, criterion)
        validate(test_loader, model, criterion, fgsm=True)
        exit()

    # lr scheduling when lr does not change
    scheduler = ReduceLROnPlateau(optimizer, mode='min',patience = 2,factor = 0.5,threshold = 0.001)
    model.cuda()

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for epoch in range(args.start_epoch, args.epochs):
        print(get_lr(optimizer))
        adjust_learning_rate(optimizer, epoch, args)
        # random sampling at each epoch for preventing overfitting
        part_tr = torch.utils.data.random_split(train_dataset, [1000, len(train_dataset)-1000])[0]
        part_te = torch.utils.data.random_split(test_dataset, [300, len(test_dataset)-300])[0]
        
        test_loader = torch.utils.data.DataLoader(
        part_te, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False)
        train_loader = torch.utils.data.DataLoader(
        part_tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

        # train for one epoch
        total_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set and fgsm_validation set
        acc1 = validate(test_loader, model, criterion).cpu()
        _ = validate(test_loader, model, criterion, fgsm=True).cpu()
        # lr sheduling by using total_loss

        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, 0, epoch)
        


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
	
    print('-' * 50)
    print('Epoch {}/{}'.format(epoch + 1, args.epochs))
    # set to fgsm epsillon
    eps=[0.007,0.02, 0.05]
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    correct_ = 0
    total_ = 0
    for idx, (images, target) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        if args.gpu is not None:
            
            # random selection fgsm epsillon
            images = images.cuda()
            target = target.cuda()
            # generate adversarial image
            
            
            a_images = fgsm_attack(model, criterion, images, target, eps[idx%3])
 
            
        
        outputs = model(images)
        _, r_preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, target)
        # get output for generate sample
        a_outputs = model(a_images)
        _, preds = torch.max(a_outputs.data, 1)

         
        
        # loss to target for adversarial sample 
        a_loss = criterion(a_outputs, target)

        # loss to original image prediction for adversarial sample
        sym_loss = criterion(a_outputs, r_preds)

        correct_ += preds.eq(target.data.view_as(preds)).long().cpu().sum()
        total_ += images.size()[0]
        total_loss = 0.7*loss + 0.3*a_loss + 0.3*sym_loss
        
        total_loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            test_acc = 100. * correct_ / float(total_)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} acc:{:.3f}%'.format(
                epoch + 1, idx * len(images), len(train_loader.dataset),
                100. * idx / len(train_loader), total_loss.item(), test_acc))
            correct_ = 0 
            total_ = 0
        running_loss += total_loss.item() * 2
        running_corrects += preds.eq(target.data.view_as(preds)).long().cpu().sum()
        
        # statistics
        

    epoch_loss = running_loss / float(len(train_loader.dataset))
    epoch_acc = running_corrects / float(len(train_loader.dataset))
    print('Training Loss: {:.4f} '
          'Acc: {:.4f}'.format(epoch_loss,
                                epoch_acc))
    return total_loss


def validate(test_loader, model, criterion, fgsm =False):
    model.eval()
    eps=[0.007, 0.02 , 0.05]
    test_loss = 0
    correct = 0
    # evaluate fgsm_validation set
    if fgsm is True:
        print("attack!")
        for idx, (images, target) in tqdm(enumerate(test_loader)):
            
            images = fgsm_attack(model, criterion, images.cuda(), target.cuda(), eps[idx%3]).cuda()
            target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            test_loss += criterion(output, target).item()
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= float(len(test_loader.dataset))
        test_acc = 100. * correct / float(len(test_loader.dataset))
        
        print('\nTest set: Average loss: {:.4f}, '
                  'Accuracy: {}/{} ({:.3f}%)\n'.format(test_loss,
                                                       correct, len(test_loader.dataset), test_acc))

        return test_acc
    # evaluate validation set
    else:
        with torch.no_grad():
            for idx, (images, target) in tqdm(enumerate(test_loader)):
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)
    
                output = model(images)
                test_loss += criterion(output, target).item()
    
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    
            test_loss /= float(len(test_loader.dataset))
            test_acc = 100. * correct / float(len(test_loader.dataset))
    
            print('\nTest set: Average loss: {:.4f}, '
                  'Accuracy: {}/{} ({:.3f}%)\n'.format(test_loss,
                                                       correct, len(test_loader.dataset), test_acc))
    
            return test_acc


def save_checkpoint(state, is_best, epoch, filename='fgsm_b6_checkpoint.pth.tar'):
    filename = str(epoch)+filename
    filename = os.path.join('/home/diml/ddrive/weights', filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '_baseline_model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
if __name__ == '__main__':
    main()
