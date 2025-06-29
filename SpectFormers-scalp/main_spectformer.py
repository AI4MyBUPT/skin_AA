# Code Adapted from GFNet(https://github.com/raoyongming/GFNet)
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import pdb

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from functools import partial
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import utils
from spectformer import SpectFormer

import warnings
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=5.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 输入格式要求：
        #   - inputs: 未经 softmax 的 logits，形状为 (batch_size, num_classes)
        #   - targets: 类别索引（hard label），形状为 (batch_size,)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 计算交叉熵
        pt = torch.exp(-ce_loss)  # 计算概率 p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Focal Loss公式
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

warnings.filterwarnings("ignore", message="Argument interpolation should be")

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str,
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.4, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-4, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=[0.9, 0.98], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.5, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.2,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=5e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=5e-4, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # 这里改了学习率

    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.4,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.2,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'SCALP'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    train_loss_history=[]
    val_loss_history=[]
    val_acc1_history=[]
    val_acc5_history=[]

    cudnn.benchmark = True

    # args.nb_classes来自自己读的
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None  # 有使用
    # 配置数据增强技术中的 Mixup 和 Cutmix 的。这两种技术都是通过在训练时将两张图片按一定比例混合，以及它们的标签，来增加模型的鲁棒性和泛化能力
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print('standard mix up') # √√√
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    else:
        print('mix up is not used')

    print(f"Creating model: {args.arch}")

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
            patch_size=16, embed_dim=384, depth=19, mlp_ratio=4, drop_path_rate=0.15,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'spectformer-b':
        model = SpectFormer(
            img_size=args.input_size, 
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.25,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    else:
        raise NotImplementedError

    if args.finetune:
        if args.finetune.startswith('https'):
            # 如果args.finetune参数以https开头，则使用torch.hub.load_state_dict_from_url函数从URL加载预训练权重
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            # 否则，使用torch.load函数从本地文件路径加载预训练权重
            checkpoint = torch.load(args.finetune, map_location='cpu')
        # 从加载的检查点（checkpoint）中提取模型权重
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()  # 获取当前模型的state_dict，这是模型参数的键值对映射
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                # 检查预训练权重中的某些特定键（例如'head.weight'、'head.bias'等）是否与新模型的权重形状不匹配。
                # 如果不匹配，则从预训练权重中删除这些键。

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        # 从预训练权重中提取位置嵌入pos_embed

        if args.arch in ['spectformer-ti', 'spectformer-xs', 'spectformer-s', 'spectformer-b']:
            num_patches = (args.input_size // 16) ** 2
        elif args.arch in ['spectformer-h-ti', 'spectformer-h-s', 'spectformer-h-b']:
            num_patches = (args.input_size // 4) ** 2
        else:
            raise NotImplementedError
                
        # 计算原始位置嵌入的大小（orig_size）和新模型的位置嵌入大小（new_size）
        num_extra_tokens = 0
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)

        scale_up_ratio = new_size / orig_size
        # class_token and dist_token are kept unchanged
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # 对位置嵌入进行 bicubic 插值以适应新的大小
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # 将插值后的位置嵌入放回预训练权重中
        checkpoint_model['pos_embed'] = pos_tokens

        # 调整复杂权重（Complex Weights）
        for name in checkpoint_model.keys():
            if 'complex_weight' in name:
                # 对于具有'complex_weight'的层，调整其权重大小以适应新的模型架构
                h, w, num_heads = checkpoint_model[name].shape[0:3] # h, w, c, 2
                origin_weight = checkpoint_model[name]
                # 计算上采样大小upsample_h和upsample_w
                upsample_h = h * new_size // orig_size
                upsample_w = upsample_h // 2 + 1
                origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
                # 对权重进行 bicubic 插值并重新调整其形状
                new_weight = torch.nn.functional.interpolate(
                    origin_weight, size=(upsample_h, upsample_w), mode='bicubic', align_corners=True).permute(0, 2, 3, 1).reshape(upsample_h, upsample_w, num_heads, 2)
                # 使用model.load_state_dict方法将调整后的预训练权重加载到模型中，strict=True表示严格匹配权重键的名称和形状
                checkpoint_model[name] = new_weight
        model.load_state_dict(checkpoint_model, strict=True)
        # 总的来说，这段代码是为了在微调（finetune）过程中将预训练模型的权重
        # 迁移到新的模型架构中，并进行必要的调整以适应新的模型大小和结构

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        # criterion = SoftTargetCrossEntropy()
        criterion = FocalLoss()
        # criterion = LabelSmoothingCrossEntropy()

    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print('损失函数=',criterion)
    # pdb.set_trace()
    teacher_model = None
    if args.distillation_type != 'none':  # 未使用
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'

    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('lr scheduler will not be updated')
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    acc_s = []  # 存储每个 epoch 的准确率
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_last.pth']
            for checkpoint_path in checkpoint_paths:
                if model_ema is not None:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
        
        if (epoch + 1) % 20 == 0: #modified 20 epoch to 100 epoch
            file_name = 'checkpoint_epoch%d.pth' % epoch
            checkpoint_path = output_dir / file_name
            if model_ema is not None:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            else:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # 画图
        acc_s.append(test_stats['acc1'])
        epoch_ch = epoch+1
        # print(range(0, epoch_ch))
        # # print(acc_s)
        plt.figure()
        plt.plot(range(0, epoch_ch), acc_s)
        plt.xlabel('epoch')
        plt.ylabel('test_accuracy')
        # # plt.show()  # 显示绘制的图形
        plt.savefig('train_gfnet_512.jpg',format='jpg')
        # 关闭 Figure
        plt.close()




        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        #########################################
        # #added by badri
        # print('Test accuracy log Acc@1', test_stats["acc1"], 'Acc@5',test_stats["acc5"])
        # print('Loss log test loss', test_stats["loss"],'train loss', train_stats["loss"])
        train_loss_history.append(train_stats["loss"])
        val_loss_history.append(test_stats["loss"])
        val_acc1_history.append(test_stats["acc1"])
        val_acc5_history.append(test_stats["acc5"])

        plt.figure()
        plt.plot(train_loss_history,label='train')
        plt.plot(val_loss_history,label='val')
        plot_loss_path = output_dir / 'loss_plot.png'
        plt.savefig(plot_loss_path)
        # 关闭 Figure
        plt.close()
        plt.figure()
        plt.plot(val_acc1_history,label='acc1')
        plt.plot(val_acc5_history,label='acc5')
        plot_acc_path = output_dir / 'acc_plot.png'
        plt.savefig(plot_acc_path)
        # 关闭 Figure
        plt.close()
        #########################################

        if max_accuracy == test_stats["acc1"]:
            checkpoint_path = output_dir / 'checkpoint_best.pth'
            if model_ema is not None:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            else:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('spectformer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
