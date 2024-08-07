import argparse
import numpy as np
import os
import random
import wandb
from collections import defaultdict
from pathlib import Path
from typing import Sized
from copy import deepcopy
from tqdm import tqdm
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


def get_args_parser():
    # others
    parser = argparse.ArgumentParser('DUSt3R training', add_help=False)
    parser.add_argument('--seed', default=777, type=int, help="Random seed")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--teacher_ckpt', default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt_new.pth", type=str, help="path to the teacher model")
    parser.add_argument('--lmd', default=30, type=float, help="kd loss weight")
    parser.add_argument('--cuda', default=7, type=int, help="cuda device")
    parser.add_argument('--ckpt', default='log/sl_kd/checkpoint-best.pth', type=str, help="resume from checkpoint")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--gt', default=True, action='store_true', help="Use ground truth depth")
    parser.add_argument('--kd_enc', default=True, type=bool)
    parser.add_argument('--kd_out', default=False, action='store_true', help="knowledge distillation (output)")
    parser.add_argument('--roma', default=1, action='store_true', help="Use RoMa")
    parser.add_argument('--encoder_only', default=False, action='store_true', help="Train only the encoder")
    parser.add_argument('--gauss_std', default=(1,3,6,9), help="Gaussian noise std")
    parser.add_argument('--asimmetric', default=False, action='store_true', help="Asymmetric loss")
    parser.add_argument('--decoder_size', default='small', type=str, help="Decoder size")
    return parser
 

def main(args):
    # SETUP
    misc.init_distributed_mode(args)
    device = f"cuda:{args.cuda}" if args.cuda >= 0 else "cpu"
    device = torch.device(device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # wandb.init(project='KDUSt3R', name=args.ckpt.split('/')[-2], config=vars(args))   

    # DATA
    TEST_DATA = f"1000 @ Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.06 0.40
    TEST_DATA += f"+ 1000 @ Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/misc/co3d_processed_xl', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.09 0.48
    TEST_DATA += f"+ 1000 @ ScanNet(split='test', ROOT='/ssd1/wenyan/scannetpp_processed', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.35 0.73
    # TEST_DATA += f"+ DL3DV(split='test', ROOT='/ssd1/sa58728/dust3r/data/DL3DV-10K', resolution=224, seed=777, gauss_std={args.gauss_std})"

    TEST_DATA += f"+ 1000@DTU(split='train', ROOT='/ssd1/sa58728/dust3r/data/dtu_processed_old', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.22 3.56
    TEST_DATA += f"+ 1000@Scannet(split='train', ROOT='/ssd1/sa58728/dust3r/data/scannet_processed', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.06 0.09
    TEST_DATA +=  f"+ 1000@BlendedMVS(split='val', ROOT='/ssd1/sa58728/dust3r/data/blendedmvs_processed', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.15 3.78
    TEST_DATA += f"+ 1000@MegaDepth(split='test', ROOT='/ssd1/sa58728/dust3r/data/megadepth_processed/', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.05 0.45

    TEST_DATA +=  f"+ 1000 @ Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/misc/co3d_processed', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.47
    # TEST_DATA += f" + 1000 @ MegaDepth(split='test', ROOT='/ssd1/sa58728/dust3r/data/megadepth_processed', resolution=224, seed=777, gauss_std={args.gauss_std})" # 0.46
    # TEST_DATA += f" + 1000 @ BlendedMVS(split='val', ROOT='/ssd1/sa58728/dust3r/data/blendedmvs_processed', resolution=224, seed=777, gauss_std={args.gauss_std})"
    # TEST_DATA += f"+ 1000 @ ScanNet(split='train', ROOT='/ssd1/wenyan/scannetpp_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, gauss_std=3)"
    TEST_DATA += f"+ 1000 @ DTU(split='test', ROOT='/ssd1/sa58728/dust3r/data/dtu_processed_new', resolution=224, seed=777, gauss_std=3)" # 0.23 3.38
    TEST_DATA += f"+ 1000 @ ARKitScenes(split='train', ROOT='/ssd1/sa58728/dust3r/data/arkitscenes_processed', resolution=224, seed=777, gauss_std=3)" # 0.12 0.13


    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in TEST_DATA.split('+')}

    # MODEL
    if args.encoder_only or args.decoder_size == 'base':
        model_dims = [384, 6, 768, 12]
    elif args.decoder_size == 'tiny':
        model_dims = [384, 6, 192, 3]
    elif args.decoder_size == 'small':
        model_dims = [384, 6, 384, 6]
    MODEL_KD = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
                output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                enc_embed_dim={}, enc_depth=12, enc_num_heads={}, dec_embed_dim={}, dec_depth=12, dec_num_heads={}, adapter=True)".format(*model_dims)
    teacher, model = build_model_enc_dec(MODEL_KD, device, args)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        
    # CRITERION
    TEST_CRITERION = f"ConfLoss(Regr3D(L21, norm_mode='avg_dis', asimmetric={args.asimmetric}), alpha=0.2) + \
                       Regr3D_ScaleShiftInv(L21, gt_scale=True, roma={args.roma}, \
                       asimmetric={args.asimmetric}, device=device)"
    test_criterion = eval(TEST_CRITERION).to(device)

    # TEST
    for test_name, testset in data_loader_test.items():
        print(test_name)
        test_one_epoch(teacher, test_criterion, testset, device, 0, args=args, teacher=teacher, test_name=test_name) #################################################################
        
        
@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int, 
                   args, teacher=None, test_name=''):
                    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for _, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        loss_tuple = loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=True, features=args.kd_enc, gt=args.gt,
                                       ret='loss', kd_enc=args.kd_enc, kd_out=args.kd_out, teacher=teacher, lmd=args.lmd, roma=args.roma)
        loss_value, loss_details = loss_tuple
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}
    
    log_dict = {"test_" + test_name+'_'+k: v for k, v in results.items()}
    # wandb.log(log_dict)
    
    return results


def build_dataset(dataset, batch_size, num_workers, test=False):
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=not (test),
                             shuffle=not (test),
                             drop_last=not (test))
    return loader


def build_model_enc_dec(model_str, device, args):

    teacher = load_model(args.teacher_ckpt, device)
    teacher.eval()
    
    model = deepcopy(teacher)
    model.to(device)
    model.eval()

    model_kd = eval(model_str)
    model_kd.to(device)
    model_kd.eval()

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model_kd.load_state_dict(ckpt['model'], strict=True)
        args.start_epoch = ckpt['epoch']
        model_kd.train()

    if args.encoder_only or args.decoder_size == 'base':
        model.patch_embed = deepcopy(model_kd.patch_embed)
        model.mask_generator = deepcopy(model_kd.mask_generator)
        model.rope = deepcopy(model_kd.rope)
        model.enc_blocks = deepcopy(model_kd.enc_blocks)
        model.enc_norm = deepcopy(model_kd.enc_norm)
        model.adapter = deepcopy(model_kd.adapter)
    else:
        model = model_kd

    return teacher, model


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
