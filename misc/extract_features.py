#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Sized

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import load_model, loss_of_one_batch
from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model

import croco.utils.misc as misc  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', 
                        default="AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")
    # dataset
    parser.add_argument('--train_dataset', type=str, help="training set")
    parser.add_argument('--test_dataset', 
        default="Co3d(split='train', ROOT='/ssd1/sa58728/dust3r/data/co3d_processed', mask_bg=False, resolution=224, seed=777) + Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/co3d_processed/', mask_bg=False, resolution=224, seed=777)",
        type=str, help="testing set")
    # training
    parser.add_argument('--seed', default=777, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=8, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=1, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    # extract features
    parser.add_argument('--save_features', default=False, action='store_true',
                        help="Extract features only, no loss computation")
    return parser

def main(args):
    misc.init_distributed_mode(args)

    print("output_dir: "+args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    args.resume = None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cpu" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


    #  dataset and loader
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=False)
                        for dataset in args.test_dataset.split('+')}

    # model
    print('Loading model')
    # model = eval(args.model)
    model = load_model(args.pretrained, device)

    model.to(device)
    # print("Model = %s" % str(model_without_ddp))

    # if args.pretrained and not args.resume:
    #     print('Loading pretrained: ', args.pretrained)
    #     ckpt = torch.load(args.pretrained, map_location=device)
    #     print(model.load_state_dict(ckpt['model'], strict=False))
    #     del ckpt  # in case it occupies memory

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)

    ############################################################################################################
        
    print(f"Start extraction")
    start_time = time.time()

    # Test on multiple datasets
    for e in range(args.epochs):
        for _, testset in data_loader_test.items():
            test_one_epoch(model, criterion=None, data_loader=testset, device=device, epoch=e, args=args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=False,
                             drop_last=False)

    print(f"{split} dataset length: ", len(loader))
    return loader



@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args):
    total, saved, same = 0, 0, 0
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for _, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        result = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=False,
            use_amp=bool(args.amp), ret=None, 
            features_only=True)
        
        if args.save_features:
            for pred_side, batch_side in zip(result, batch): # for each view
                feats, pts = pred_side
                for f, p, i in zip(feats, pts, batch_side['img_path']): # for each sample in the batch
                    total += 1
                    f = f.cpu().numpy()
                    if os.path.exists(i.replace('jpg','npy')):
                        t = np.load(i.replace('jpg','npy'))
                        if np.array_equal(t,f):
                            same += 1
                            continue

                    np.save(i.replace('jpg','npy'), f) # save features
                    saved += 1


    print(f"Saved {saved} features, {same} already exist, {total} total")
    return


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
