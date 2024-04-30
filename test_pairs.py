import argparse
import numpy as np
import os, json
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
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch, load_model
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy, to_cpu, collate_with_cat
import open3d as o3d


TEST_DATA = "Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', resolution=224, seed=777, gaussian_frames=True)" # Unseen scenes
TEST_DATA += " + ScanNet(split='test', ROOT='/ssd1/wenyan/scannetpp_processed', resolution=224, seed=777, gaussian_frames=True)" # Unseen scenes
TEST_DATA += " + DL3DV(split='test', ROOT='/ssd1/sa58728/dust3r/data/DL3DV-10K', resolution=224, seed=777, gaussian_frames=True)" # Unseen scenes

MODEL_KD = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
            output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
            enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, adapter=True)"
CKPT = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
CKPT_KD = None # "checkpoints/DUSt3R_ViTSmall_BaseDecoder_512_dpt_kd.pth"
TEST_CRITERION = "ConfLoss(Regr3D(L21, norm_mode='avg_dis', kd=True), alpha=0.2) + Regr3D_ScaleShiftInv(L21, gt_scale=True, kd=True)"

TEST_DIR = 'test'
TEST_DATASETS = ['Co3D', 'ScanNet++', 'DL3DV-10K']
STUDENT = True

device = 'cuda:7'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

def get_args_parser():
    parser = argparse.ArgumentParser('DUSt3R training', add_help=False)
    parser.add_argument('--seed', default=777, type=int, help="Random seed")
    parser.add_argument('--model', default=MODEL_KD, type=str, help="string containing the model to build")
    parser.add_argument('--test_criterion', default=TEST_CRITERION, type=str, help="test criterion")
    parser.add_argument('--teacher_path', default=CKPT, type=str, help="path to the teacher model")
    parser.add_argument('--lmd', default=10, type=float, help="kd loss weight")
    parser.add_argument('--output_dir', default='./log/train/', type=str, help="path where to save the output")
    parser.add_argument('--cuda', default=7, type=int, help="cuda device")
    parser.add_argument('--ckpt', default='/home/sa58728/dust3r/log/train_2/checkpoint-best.pth', type=str, help="resume from checkpoint") # "log/ckpt/iter_24750.pth"
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--kd', default=True, type=bool)
    parser.add_argument('--kd_out', default=True, action='store_true', help="knowledge distillation (output)")

    return parser


def main(args):
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # model and criterion
    teacher, model = load_pretrained(args.model, args.teacher_path, args.ckpt, device)
    test_criterion = eval(args.test_criterion or args.criterion).to(device)
    model.to(device)

    for d in TEST_DATASETS:
        data_path = f'{TEST_DIR}/{d}'
        for scene in os.listdir(data_path):
            if scene.startswith('.'):
                continue
            img_path = f'{data_path}/{scene}/'
            print(img_path)
            # load_images can take a list of images or a directory
            images = load_images(img_path, size=512)
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

            with torch.no_grad():
                result = loss_of_one_batch(collate_with_cat(pairs), model, test_criterion, device,
                                        symmetrize_batch=True, features=True,
                                        kd=args.kd, kd_out=args.kd_out, teacher=teacher, lmd=args.lmd)
                
            result = to_cpu(result)
            loss_value, loss_details = result['loss']  # criterion returns two values
            loss_details['loss'] = loss_value.item()
            print(f"Details: {loss_details}")

            scene = global_aligner(result, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
            _ = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
            # retrieve useful values from scene:
            imgs = scene.imgs
            pts3d = scene.get_pts3d()
            pts3d = to_numpy(pts3d)
            pts_4_3dgs = np.concatenate([item.reshape(-1, 3) for item in pts3d])
            color_4_3dgs = np.concatenate([item.reshape(-1, 3) for item in imgs])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_4_3dgs)
            pcd.colors = o3d.utility.Vector3dVector(color_4_3dgs)
            if STUDENT:
                o3d.io.write_point_cloud(f"{img_path}kd.ply", pcd)
                json.dump(loss_details, open(f"{img_path}loss.json", 'w'))
            else:
                o3d.io.write_point_cloud(f"{img_path}baseline.ply", pcd)

def load_pretrained(model_kd, teacher_path, model_kd_path, device):
    teacher = load_model(teacher_path, device)
    teacher.eval()
    if not STUDENT:
        return teacher, teacher
    
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
