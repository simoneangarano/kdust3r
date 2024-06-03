import os
import torch
from dust3r.inference import load_model, loss_of_one_batch
from dust3r.datasets.scannet import Scannet
from dust3r.utils.image import denormalize
from torchvision.utils import save_image
from dust3r.losses import *  # noqa: F401, needed when loading the model
import json
import numpy as np
from dust3r.utils.device import to_cuda
from einops import rearrange

class Args:
    def __init__(self):
        self.image_size = 224
        self.weights = 'checkpoints/519_dim64_cosloss_halfres_cross_dust_lseg_multiscale/checkpoint-80.pth'
        self.lseg_weight = 'checkpoints/demo_e200.ckpt'
        self.device = 'cuda'
        self.criterion = 'GaussianLoss()'
        self.batch_size = 1

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

args = Args()

# load data(get 20 views, seen scene but unseen view)
dataset = Scannet(split='test', ROOT="data/scannet_processed", resolution=args.image_size, num_views=20)
# criterion
criterion = eval(args.criterion).to(args.device)
test_scene_folder = 'data/test_scene_split'


# get views
# set scene id
scene_id = 'scene0687_00'
llffhold = 8

all_views = os.listdir(os.path.join(test_scene_folder, scene_id, 'images'))
all_views = sorted(all_views, key=lambda x: int(x.split('.')[0]))
train_views = [view for idx, view in enumerate(all_views) if (idx % llffhold != 1) and (idx % llffhold != 3)]
test_views = [view for idx, view in enumerate(all_views) if (idx % llffhold == 1) or (idx % llffhold == 3)]
print(f"test_views: {test_views}")
train_views = [int(x.split('.')[0]) for x in train_views]
# test_views = [int(x.split('.')[0]) for x in test_views]
test_views = train_views


# depth metric on train views
Total_rel_err = 0
Total_tau = 0
with torch.no_grad():
    for i, train_view in enumerate(train_views):
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