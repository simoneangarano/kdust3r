import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Sized
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch, load_model

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
from RoMa.roma import roma_outdoor

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

TRAIN_DATA = "100000 @ Co3d(split='train', ROOT='/ssd1/wenyan/co3d_2_cat_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, gaussian_frames=True)"
TRAIN_DATA += "+ 100000 @ ScanNet(split='train', ROOT='/ssd1/wenyan/scannetpp_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, gaussian_frames=True)"
TRAIN_DATA += "+ 100000 @ DL3DV(split='train', ROOT='/ssd1/sa58728/dust3r/data/DL3DV-10K', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, gaussian_frames=True)"
# TRAIN_DATA += " + 70000 @ MegaDepth(split='train', ROOT='/ssd1/sa58728/dust3r/data/MegaDepth_v1', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, gaussian_frames=True)"

TEST_DATA =  "1000 @ Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', resolution=224, seed=777, gaussian_frames=True)" # Unseen scenes
TEST_DATA += " + 1000 @ ScanNet(split='test', ROOT='/ssd1/wenyan/scannetpp_processed', resolution=224, seed=777, gaussian_frames=True)" # Unseen scenes
TEST_DATA += " + 1000 @ DL3DV(split='test', ROOT='/ssd1/sa58728/dust3r/data/DL3DV-10K', resolution=224, seed=777, gaussian_frames=True)" # Unseen scenes
# TEST_DATA += " + 1000 @ MegaDepth(split='test', ROOT='/ssd1/sa58728/dust3r/data/MegaDepth_v1', resolution=224, seed=777, gaussian_frames=True)" # Unseen scenes

# TEST_DATA += " + 1000 @ Co3d(split='test', ROOT='/ssd1/wenyan/co3d_2_cat_processed', resolution=224, seed=777, gaussian_frames=True)" # Seen scenes

MODEL = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
         output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
         enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)"
MODEL_KD = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
            output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
            enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, adapter=True)"
CKPT = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
CKPT_KD = None # "checkpoints/DUSt3R_ViTSmall_BaseDecoder_512_dpt_kd.pth"

TRAIN_CRITERION = "ConfLoss(Regr3D(L21, norm_mode='avg_dis', kd=True), alpha=0.2)"
TEST_CRITERION = "ConfLoss(Regr3D(L21, norm_mode='avg_dis', kd=True), alpha=0.2) + Regr3D_ScaleShiftInv(L21, gt_scale=True, kd=True)"

def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default=MODEL_KD,
                        type=str, help="string containing the model to build")
    parser.add_argument('--train_criterion', default=TRAIN_CRITERION,
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=TEST_CRITERION, type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', default=TRAIN_DATA, type=str, help="training set")
    parser.add_argument('--test_dataset', default=TEST_DATA, type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=777, type=int, help="Random seed")
    parser.add_argument('--epochs', default=100, type=int, help="Maximum number of epochs for the scheduler")

    parser.add_argument('--weight_decay', type=float, default=0.00005, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.00015, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

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
    parser.add_argument('--print_freq', default=100, type=int,
                        help='frequence (number of iterations) to print infos while training')
    
    # test
    parser.add_argument('--test', default=False, action='store_true', help="test only flag")
    parser.add_argument('--kd', default=True, action='store_true', help="knowledge distillation (features)")
    parser.add_argument('--kd_out', default=True, action='store_true', help="knowledge distillation (output)")
    parser.add_argument('--teacher_path', default=CKPT, type=str, help="path to the teacher model")
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--lmd', default=10, type=float, help="kd loss weight")
    parser.add_argument('--output_dir', default='./log/roma/', type=str, help="path where to save the output")
    parser.add_argument('--cuda', default=-1, type=int, help="cuda device")
    parser.add_argument('--pretrained', default=False, help='path of a starting checkpoint') # CKPT_KD
    parser.add_argument('--ckpt', default=None, type=str, help="resume from checkpoint") # 'log/train_10_1%/checkpoint-1.pth'
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int, help="Accumulate gradient iterations")
    parser.add_argument('--roma', default=True, action='store_true', help="Use RoMa")

    return parser


def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    args.resume = None

    device = f"cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True
    # cudnn.deterministic = True

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    print('Building test dataset {:s}'.format(args.train_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # model and criterion
    if args.test or args.pretrained:
        teacher, model = load_pretrained(args.model, args.teacher_path, args.ckpt, device)
    else:
        teacher, model = build_model_enc_dec(args.model, device, args)

    if args.roma:
        roma_model = roma_outdoor(device=device, coarse_res=224, upsample_res=224)
    else:
        roma_model = None

    train_criterion = eval(args.train_criterion).to(device)
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], static_graph=True)
        model_without_ddp = model.module
        train_modules = [model_without_ddp.patch_embed, model_without_ddp.mask_generator, model_without_ddp.rope, model_without_ddp.enc_blocks, model_without_ddp.enc_norm]

        if args.roma:
            roma_model = torch.nn.parallel.DistributedDataParallel(roma_model, device_ids=[args.gpu], static_graph=True)

    else:
        model_without_ddp = model
        train_modules = [model_without_ddp.patch_embed, model_without_ddp.mask_generator, model_without_ddp.rope, model_without_ddp.enc_blocks, model_without_ddp.enc_norm]

    if hasattr(model_without_ddp, 'adapter') and model_without_ddp.adapter is not None:
        train_modules.append(model_without_ddp.adapter)
    else:
        print("No adapter found in the model")

    train_params = torch.nn.ParameterList([p for m in train_modules for p in m.parameters()])
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay) #, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(**{f'train_{k}': v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update({test_name+'_'+k: v for k, v in test_stats[test_name].items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)

    best_so_far = misc.load_model(args=args, model=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {'train_iter': 0}
    for epoch in range(args.start_epoch, args.epochs):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch-1, 'last', best_so_far)

        # Test on multiple datasets
        new_best = False
        if ((epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0) or args.test) and log_writer is not None:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model_without_ddp, test_criterion, testset,
                                       device, epoch, log_writer=log_writer, args=args, prefix=test_name,
                                       teacher=teacher, roma_model=roma_model, curr_step=train_stats['train_iter'])
                test_stats[test_name] = stats

            # Save best of all
            if stats['loss_med'] < best_so_far:
                best_so_far = stats['loss_med']
                new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch-1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch-1, 'best', best_so_far)
        if epoch >= args.epochs or args.test:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, teacher=teacher, roma_model=roma_model,
            args=args, train_params=train_params, data_loader_test=data_loader_test,
            test_criterion=test_criterion, best_so_far=best_so_far)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model if isinstance(model, dict) else model.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, args, teacher=None, roma_model=None,
                    log_writer=None, train_params=None, data_loader_test=None, test_criterion=None, best_so_far=inf):
    assert torch.backends.cuda.matmul.allow_tf32 == True
    model_without_ddp = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model = set_trainable(model)
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.1e}\n>'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True, features=args.kd,
                                       use_amp=bool(args.amp), ret='loss', kd=args.kd, kd_out=args.kd_out,
                                       teacher=teacher, lmd=args.lmd, roma_model=roma_model)
        loss, loss_details = loss_tuple  # criterion returns two values
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=train_params,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        # metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step) % accum_iter == 0 and ((data_iter_step) % (accum_iter * args.print_freq)) == 0:
            # loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            epoch_1000x = int(epoch * len(data_loader) + data_iter_step) // accum_iter

            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            for name, _ in loss_details.items():
                log_writer.add_scalar('train_'+name, getattr(metric_logger, name).avg, epoch_1000x)

        if do_test_now(data_iter_step, accum_iter) and log_writer is not None:
            new_best = False
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                print(test_name)
                stats = test_one_epoch(model_without_ddp, test_criterion, testset,
                                       device, epoch, log_writer=log_writer, args=args, prefix=test_name,
                                       teacher=teacher, roma_model=roma_model, curr_step=epoch_1000x)
                test_stats[test_name] = stats

            # Save best of all
            if stats['loss_med'] < best_so_far:
                best_so_far = stats['loss_med']
                new_best = True

            if new_best:
                misc.save_model(args=args, model=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch-1, fname='best', best_so_far=best_so_far)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['train_iter'] = epoch_1000x
    return stats


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test', 
                   teacher=None, roma_model=None, curr_step=0):
                    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]\n>'.format(epoch)

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for _, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True, features=args.kd,
                                       use_amp=bool(args.amp), ret='loss', 
                                       kd=args.kd, kd_out=args.kd_out, roma_model=roma_model,
                                       teacher=teacher, lmd=args.lmd)
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    epoch_1000x = curr_step

    for name, val in results.items():
        log_writer.add_scalar(prefix+'_'+name, val, epoch_1000x)
    return results


def build_model_enc_dec(model_str, device, args):
    teacher = load_model("checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", device)
    teacher.eval()

    if "x" in args.output_dir:
        print("Using pretrained Dust3R")
        model = load_model("checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", device)
    else:
        print("Training from scratch")
        model = eval(model_str)

    model.to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        print(model.load_state_dict(ckpt['model'], strict=True))
        args.start_epoch = ckpt['epoch']
        model.train()

    module_list = ['decoder_embed', 'dec_blocks', 'dec_norm', 'dec_blocks2', 'downstream_head1', 'downstream_head2']
    for m in module_list:
        getattr(model, m).load_state_dict(getattr(teacher, m).state_dict(), strict=True)
        getattr(model, m).eval()
    model.mask_token = teacher.mask_token

    return teacher, model


def set_trainable(model):
    model.train()
    module_list = ['decoder_embed', 'dec_blocks', 'dec_norm', 'dec_blocks2', 'downstream_head1', 'downstream_head2']
    for m in module_list:
        if hasattr(model, m):
            getattr(model, m).eval()
        else:
            getattr(model.module, m).eval()
    if hasattr(model, 'mask_token'):
        model.mask_token.requires_grad = False
    else:
        model.module.mask_token.requires_grad = False
    return model


def load_pretrained(model_kd, teacher_path, model_kd_path, device):
    print("Loading student model from: ", model_kd_path)
    teacher = load_model(teacher_path, device)
    teacher.eval()

    model = deepcopy(teacher)
    model.to(device)
    model.eval()

    model_kd = eval(model_kd)
    model_kd.to(device)
    model_kd.eval()

    ckpt = torch.load(model_kd_path, map_location=device)
    try:
        print(model_kd.load_state_dict(ckpt['model'], strict=True))
        args.start_epoch = ckpt['epoch']
    except:
        print(model_kd.load_state_dict(ckpt, strict=True))
    del ckpt  # in case it occupies memory

    model.patch_embed = deepcopy(model_kd.patch_embed)
    model.mask_generator = deepcopy(model_kd.mask_generator)
    model.rope = deepcopy(model_kd.rope)
    model.enc_blocks = deepcopy(model_kd.enc_blocks)
    model.enc_norm = deepcopy(model_kd.enc_norm)
    model.adapter = deepcopy(model_kd.adapter)

    return teacher, model


def do_test_now(data_iter_step, accum_iter):
    output = (data_iter_step) % accum_iter == 0 and ((data_iter_step) % (accum_iter * 2500)) == 0
    return output

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
