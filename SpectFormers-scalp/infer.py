import argparse
import datetime
import csv

import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
from sklearn.metrics import confusion_matrix
# from openpyxl import Workbook
import sys
from pathlib import Path
import os
import shutil
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import utils
from functools import partial

from spectformer import SpectFormer, _cfg

def get_args_parser():
    parser = argparse.ArgumentParser('spectformer evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--arch', default='spectformer-xs', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=512, type=int, help='images input size')
    parser.add_argument('--data-path', default='/data/yike/cyl/data_test/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'SCALP'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model-path', default='/data/yike/cyl/SpectFormers-main/vanilla_architecture/logs/spectformer-xs-3-512/checkpoint_best.pth', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):

    cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.arch == 'spectformer-xs':
        model = SpectFormer(
            img_size=args.input_size, 
            patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'spectformer-ti':
        model = SpectFormer(
            img_size=args.input_size, 
            patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'spectformer-s':
        model = SpectFormer(
            img_size=args.input_size, 
            patch_size=16, embed_dim=384, depth=19, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'spectformer-b':
        model = SpectFormer(
            img_size=args.input_size, 
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    else:
        raise NotImplementedError

    model_path = args.model_path
    model.default_cfg = _cfg()

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    print('## model has been successfully loaded')

    model = model #.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss() #.cuda()
    validate(data_loader_val, model, criterion)

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    model.eval()
    logSigmoid_fun = nn.LogSoftmax()  
    Softmax_fun = nn.Softmax()
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')
    test_num = len(val_loader)

    print("test_num={}".format(test_num))
    with torch.no_grad():
        end = time.time()
        predictlist=[]
        labellist=[]
        result_list=[]   # 打印
        k=0  
        # with open('output.txt', 'w') as f:
        #     sys.stdout = f
        for i, (images, target) in enumerate(val_loader):

            images = images #.cuda()
            # images_arr =images[0].item()
            # print('images={}'.format(images))
            target = target #.cuda()
            # print("target={}".format(target))
            num_label = target.item()
            labellist.append(num_label)
            # print(labellist)

            # compute output
            output = model(images)
            predict_y = torch.max(output, dim=1)[1]
            num_predict = predict_y.item()
            predictlist.append(num_predict)

            aa = Softmax_fun(output[0,:len(val_loader.dataset.classes)])#.unsqueeze(0)  #len(val_loader.dataset.classes)=11classnum

            path, label = val_loader.dataset.samples[i]
            file_name = os.path.basename(path)
            label_name = val_loader.dataset.classes[label]
            result_list.append(i+1)
            result_list.append(label_name)
            result_list.append(file_name)
            for j,value in enumerate(aa):
                resultj = val_loader.dataset.classes[j]
                pre = value.item()
                result_list.append(resultj)
                result_list.append(pre)

            loss = criterion(output, target)

            
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))

            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))
            # print("top1={}".format(top1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)



        # sys.stdout = sys.__stdout__
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
              .format(top1=top1, top2=top2))
        print(val_loader.dataset.classes)
        # print('labellist={}'.format(labellist))
        # print('predictlist={}'.format(predictlist))
        c = confusion_matrix(labellist,predictlist)
        print(c)
        num_classes = c.shape[0]

        class_accuracies = np.diag(c) / np.sum(c, axis=1)
        # label_mapping = {
        #     'AA': 'Alopecia areata',
        #     'AGA': 'Androgenetic alopecia',
        #     'Nevus': 'Nevus',
        #     'Pso': 'Psoriasis',
        #     'SA': 'Scarring alopecia',
        #     'SD': 'Seborrheic dermatitis',
        #     'SK': 'Seborrheic keratosis',
        #     'Scalp': 'Scalp',
        #     'TTM': 'Trichotillomania',
        #     'Warts': 'Warts',
        #     'skin': 'Non-scalp skin'
        #     }
        # order = ['Alopecia areata', 'Androgenetic alopecia', 'Scarring alopecia', 'Trichotillomania', 
        #         'Psoriasis', 'Seborrheic dermatitis', 'Nevus', 'Seborrheic keratosis', 'Warts', 'Non-scalp skin']
        # for label in order:
        #     index = list(label_mapping.values()).index(label)
        #     acc_class = class_accuracies[index]
        #     print(f'{label}: {acc_class:.3f}')

        for i, acc_class in enumerate(class_accuracies):
            print(f'acc_class of class {i} ({val_loader.dataset.classes[i]}): {acc_class:.3f}')


    return top1.avg
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser('spectformer evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
