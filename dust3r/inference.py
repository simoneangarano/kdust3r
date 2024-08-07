# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch, numpy as np
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf

H, W = 224, 224
ROMA_STD = torch.Tensor([0.229, 0.224, 0.225])[:,None,None]
ROMA_MEAN = torch.Tensor([0.485, 0.456, 0.406])[:,None,None]

def load_model(model_path, device):
    print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    print(f"instantiating : {args}")
    net = eval(args)
    print(net.load_state_dict(ckpt['model'], strict=False))
    return net.to(device)


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None, 
                      return_times=False, features_only=False, features=False, gt=True,
                      kd_enc=False, kd_out=False, teacher=None, lmd=1, lmd_out=1, criterion_kd=torch.nn.MSELoss(),
                      roma=None):
    view1, view2 = batch
    for view in batch:
        for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres index'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        outs = model(view1, view2, return_times=return_times, features_only=features_only, features=features)
        if return_times:
            pred1, pred2, times = outs
        else:
            pred1, pred2 = outs

        if features_only or not gt:
            loss = 0
        else:
            # loss is supposed to be symmetric
            with torch.cuda.amp.autocast(enabled=False):
                if criterion is not None:
                    loss = criterion(view1, view2, pred1, pred2)
                    loss_tot = loss[0].clone()
                    loss_dict = loss[1].copy()
                else:
                    loss = None
                    loss_tot = 0
                    loss_dict = {}

    if kd_enc or kd_out:
        with torch.no_grad():
            teacher_outs = teacher(view1, view2, return_times=return_times, features_only=features_only, features=features)

        if kd_out:
            teach1, teach2 = teacher_outs
            teach1['img'] = view1['img']
            teach2['img'] = view2['img']
            teach1['index'] = view1['index']
            teach2['index'] = view2['index']
            # loss is supposed to be symmetric
            with torch.cuda.amp.autocast(enabled=False):
                loss_kd_out = criterion(teach1, teach2, pred1, pred2, kd=True) if criterion is not None else None
                    
            loss_tot += loss_kd_out[0].clone() * lmd_out
            loss_dict['kd_out'] = loss_kd_out[0].item()

        loss_kd = 0
        if kd_enc:
            for pred_side, teacher_side in zip(outs, teacher_outs): # for each view
                pred_feats = pred_side['features']
                target_feats = teacher_side['features']
                loss_kd += criterion_kd(pred_feats, target_feats) / 2
            loss_tot += loss_kd * lmd
            
            loss_dict['kd'] = loss_kd.item()

        loss = (loss_tot, loss_dict)

    if roma and loss_dict['roma_mae'] > 0:
        loss_tot += loss_dict['roma_mae'] * roma
        loss = (loss_tot, loss_dict)
        loss_dict['roma_mae'] = loss_dict['roma_mae'].item()
        loss_dict['roma_mse'] = loss_dict['roma_mse'].item()

    # if debug:
    #     loss_dict['pred'] = outs

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    if kd_enc or kd_out:
        result['teacher_outs'] = teacher_outs

    if return_times:
        return result, times
    elif features_only:
        return outs
    return result[ret] if ret else result


@torch.no_grad()
def inference(pairs, model, device, batch_size=8, return_times=False, features_only=False):
    print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i+batch_size]), model, None, device, return_times=return_times, features_only=features_only)
        if return_times:
            res, times = res
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    torch.cuda.empty_cache()
    return result, times if return_times else result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
