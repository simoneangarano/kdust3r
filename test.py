import argparse
import numpy as np
import os
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


TEST_DATA = "1000 @ Co3d(split='test', ROOT='/ssd1/wenyan/co3d_2_cat_processed', resolution=224, seed=777)"
TEST_DATA += " + 1000 @ Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', resolution=224, seed=777)"
TEST_DATA += " + 1000 @ ScanNet(split='train', ROOT='/ssd1/wenyan/scannetpp_processed', resolution=224, seed=777)"
MODEL_KD = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
            output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
            enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, adapter=True)"
CKPT = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
CKPT_KD = None # "checkpoints/DUSt3R_ViTSmall_BaseDecoder_512_dpt_kd.pth"
TEST_CRITERION = "Regr3D_ScaleShiftInv(L21, gt_scale=True)"


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default=MODEL_KD, type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint') # CKPT_KD
    parser.add_argument('--test_criterion', default=TEST_CRITERION, type=str, help="test criterion")
    # dataset
    parser.add_argument('--test_dataset', default=TEST_DATA, type=str, help="testing set")
    parser.add_argument('--seed', default=777, type=int, help="Random seed")
    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=100, type=int,
                        help='frequence (number of iterations) to print infos while training')
    parser.add_argument('--teacher_path', default=CKPT, type=str, help="path to the teacher model")

    parser.add_argument('--lmd', default=10, type=float, help="kd loss weight")
    parser.add_argument('--output_dir', default='./log/train/', type=str, help="path where to save the output")
    parser.add_argument('--cuda', default=7, type=int, help="cuda device")
    parser.add_argument('--ckpt', default='/home/sa58728/dust3r/log/co3d_scannet_10/checkpoint-best.pth', type=str, help="resume from checkpoint") # "log/ckpt/iter_24750.pth"
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int, help="Accumulate gradient iterations")
    parser.add_argument('--kd', default=True, type=bool)

    return parser


def main(args):
    misc.init_distributed_mode(args)
    # global_rank = misc.get_rank()

    print("output_dir: "+args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # model and criterion
    teacher, model = load_pretrained(args.model, args.teacher_path, args.ckpt, device)

    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)
    # print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)

    log_writer = SummaryWriter(log_dir=args.output_dir)

    train_stats = {'train_iter': 0}

    # Test on multiple datasets
    for test_name, testset in data_loader_test.items():
        test_one_epoch(teacher, test_criterion, testset,
                       device, 0, log_writer=log_writer, args=args, prefix=test_name,
                       kd=args.kd, teacher=teacher, features=args.kd, curr_step=train_stats['train_iter'])
        test_one_epoch(model, test_criterion, testset,
                       device, 0, log_writer=log_writer, args=args, prefix=test_name,
                       kd=args.kd, teacher=teacher, features=args.kd, curr_step=train_stats['train_iter'])
        
@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test', kd=False, teacher=None, features=False, curr_step=0):
                    
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
                                       symmetrize_batch=True, features=features,
                                       ret='loss', kd=kd, teacher=teacher, lmd=args.lmd)
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    return results


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


def load_pretrained(model_kd, teacher_path, model_kd_path, device):
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


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
