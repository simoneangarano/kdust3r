import os
import torch
from dust3r.inference import load_model, loss_of_one_batch
from dust3r.datasets.scannet import Scannet
from dust3r.losses import *  # noqa: F401, needed when loading the model
import numpy as np
from dust3r.utils.device import to_cuda

import croco.utils.misc as misc  # noqa
import random
import torch.backends.cudnn as cudnn
import argparse


def find_nearest_view(test_viewid, train_views):
    nearest_numbers = sorted(train_views, key=lambda x: abs(x - test_viewid))[:2]
    return nearest_numbers

def convert_to_batch(view):
    # convert to batch
    data_list = [view]
    batch_data = {key: [] for key in data_list[0]}
    for sample in data_list:
        for key, value in sample.items():
            batch_data[key].append(value)
    for key, value in batch_data.items():
        if isinstance(value[0], (int, float)):
            batch_data[key] = torch.tensor(value)
        elif isinstance(value[0], torch.Tensor):
            batch_data[key] = torch.stack(value)
        elif isinstance(value[0], np.ndarray):
            batch_data[key] = torch.tensor(value)
        else:
            batch_data[key] = value
    return batch_data

def calculate_depth_metrics(pred_depth, gt_depth):
    # create mask
    mask = gt_depth > 0
    
    # apply mask
    gt_depth_masked = gt_depth[mask]
    pred_depth_masked = pred_depth[mask]
    
    # calculate the scale
    scale = torch.median(gt_depth_masked) / torch.median(pred_depth_masked)
    
    # scale the pred depth
    pred_depth_masked *= scale
    
    # calculate the metrics
    rel_err = torch.abs(gt_depth_masked - pred_depth_masked) / gt_depth_masked
    # tau(mean of max(pred/gt, gt/pred) < 1.03)
    tau = ((torch.max(pred_depth_masked / gt_depth_masked, gt_depth_masked / pred_depth_masked) < 1.03).float()).mean()
    return rel_err.mean() * 100, tau * 100


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

def get_args_parser():
    # others
    parser = argparse.ArgumentParser('DUSt3R training', add_help=False)
    parser.add_argument('--seed', default=777, type=int, help="Random seed")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--criterion', default='GaussianLoss()', type=str, help="criterion")
    parser.add_argument('--image_size', default=224, type=int, help="image size")

    parser.add_argument('--teacher_ckpt', default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", type=str, help="path to the teacher model")
    parser.add_argument('--lmd', default=10, type=float, help="kd loss weight")
    parser.add_argument('--cuda', default=4, type=int, help="cuda device")
    parser.add_argument('--ckpt', default='log/gauss3_roma1000_init_new/checkpoint-best.pth', type=str, help="resume from checkpoint")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--kd_enc', default=True, type=bool)
    parser.add_argument('--kd_out', default=True, action='store_true', help="knowledge distillation (output)")
    parser.add_argument('--roma', default=1, action='store_true', help="Use RoMa")
    parser.add_argument('--encoder_only', default=True, action='store_true', help="Train only the encoder")
    parser.add_argument('--gauss_std', default=(1,3,6,9), help="Gaussian noise std")
    parser.add_argument('--asimmetric', default=False, action='store_true', help="Asymmetric loss")
    parser.add_argument('--decoder_size', default='base', type=str, help="Decoder size")
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

    # load data(get 20 views, seen scene but unseen view)
    dataset = Scannet(split='test', ROOT="/ssd1/sa58728/dust3r/data/scannet_processed", resolution=args.image_size, num_views=20)
    # criterion
    # criterion = eval(args.criterion).to(args.device)
    # test_scene_folder = '/ssd1/sa58728/dust3r/data/scannet_processed/test_scene_split'

    # # get views
    # # set scene id
    # scene_id = 'scene0687_00'
    # llffhold = 8

    # all_views = os.listdir(os.path.join(test_scene_folder, scene_id, 'images'))
    # all_views = sorted(all_views, key=lambda x: int(x.split('.')[0]))
    # train_views = [view for idx, view in enumerate(all_views) if (idx % llffhold != 1) and (idx % llffhold != 3)]
    # test_views = [view for idx, view in enumerate(all_views) if (idx % llffhold == 1) or (idx % llffhold == 3)]
    # print(f"test_views: {test_views}")
    # train_views = [int(x.split('.')[0]) for x in train_views]
    # # test_views = [int(x.split('.')[0]) for x in test_views]
    # test_views = train_views

    # MODEL
    if args.encoder_only or args.decoder_size == 'base':
        model_dims = [384, 6, 768, 12]
    elif args.decoder_size == 'tiny':
        model_dims = [384, 6, 192, 3]
    MODEL_KD = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
                output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                enc_embed_dim={}, enc_depth=12, enc_num_heads={}, dec_embed_dim={}, dec_depth=12, dec_num_heads={}, adapter=True)".format(*model_dims)
    teacher, model = build_model_enc_dec(MODEL_KD, device, args)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)

    # depth metric on train views
    Total_rel_err = 0
    Total_tau = 0
    with torch.no_grad():
        for i, train_view in enumerate(dataset):
            # find nearest view(itself and another view)
            source_view1_id, source_view2_id = find_nearest_view(train_view, train_views)
            source_view1 = dataset.get_test_views(scene_id, source_view1_id, args.image_size)
            source_view1 = convert_to_batch(source_view1)
            source_view2 = dataset.get_test_views(scene_id, source_view2_id, args.image_size)
            source_view2 = convert_to_batch(source_view2)
            source_view1 = to_cuda(source_view1)
            source_view2 = to_cuda(source_view2)
            
            # inference
            data_batch = [source_view1, source_view2]
            output = loss_of_one_batch(data_batch, model, None, args.device)
            
            # get pred depth
            pred1_depth = output['pred1']['pts3d'][..., 2]
            pred2_depth = output['pred2']['pts3d_in_other_view'][..., 2]
            
            # get gt depth
            gt1_depth = source_view1['depthmap']
            gt2_depth = source_view2['depthmap']
            
            # create mask
            mask1 = gt1_depth > 0
            mask2 = gt2_depth > 0
            
            # apply mask
            pred1_depth_masked = pred1_depth[mask1]
            pred2_depth_masked = pred2_depth[mask2]
            gt1_depth_masked = gt1_depth[mask1]
            gt2_depth_masked = gt2_depth[mask2]
            
            # calculate the scale
            scale1 = torch.median(gt1_depth_masked) / torch.median(pred1_depth_masked)
            scale2 = torch.median(gt2_depth_masked) / torch.median(pred2_depth_masked)
            
            # scale the pred depth
            pred1_depth_masked *= scale1
            pred2_depth_masked *= scale2
            
            # calculate the metrics
            rel_err1 = torch.abs(pred1_depth_masked - gt1_depth_masked) / gt1_depth_masked
            rel_err2 = torch.abs(pred2_depth_masked - gt2_depth_masked) / gt2_depth_masked
            # tau(mean of max(pred/gt, gt/pred) < 1.03)
            tau1 = ((torch.max(pred1_depth_masked / gt1_depth_masked, gt1_depth_masked / pred1_depth_masked) < 1.03).float()).mean()
            tau2 = ((torch.max(pred2_depth_masked / gt2_depth_masked, gt2_depth_masked / pred2_depth_masked) < 1.03).float()).mean()
            
            # print
            print(f"train_view: {train_view}, "
                f"rel_err1: {rel_err1.mean()}, "
                f"rel_err2: {rel_err2.mean()}, "
                f"tau1: {tau1}, "
                f"tau2: {tau2}")
            
            # sum
            Total_rel_err += (rel_err1.mean() + rel_err2.mean()) / 2 * 100
            Total_tau += (tau1 + tau2) / 2 * 100
            torch.cuda.empty_cache()
            del pred1_depth_masked, pred2_depth_masked, gt1_depth_masked, gt2_depth_masked


    Total_rel_err /= len(train_views)
    Total_tau /= len(train_views)

    print(f"mean_rel_err: {Total_rel_err}, mean_tau: {Total_tau}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)