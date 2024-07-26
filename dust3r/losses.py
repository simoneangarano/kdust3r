# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn
import numpy as np

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
from RoMa.roma import roma_outdoor

H, W = 224, 224
ROMA_MEAN = torch.Tensor([0.485, 0.456, 0.406])[:,None,None]
ROMA_STD = torch.Tensor([0.229, 0.224, 0.225])[:,None,None]

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class LLoss (nn.Module):
    """ L-norm loss
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim-1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), f'{criterion} is not a proper criterion!' #+bb()
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = 'none'  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, roma=None, roma_thr=0.5, conf_thr=2,
                 debug=False, asimmetric=False, device='cuda'):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.roma = roma
        self.roma_thr = roma_thr
        self.conf_thr = conf_thr
        self.debug = debug
        self.asimmetric = asimmetric
        if self.roma:
            self.roma = roma_outdoor(device=device, coarse_res=224, upsample_res=224)

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None, kd=False):
        if kd:
            valid1 = valid2 = torch.ones_like(gt1['pts3d'][..., 0], dtype=torch.bool)

            pr_pts1 = get_pred_pts3d(gt=None, pred=pred1, use_pose=False)
            pr_pts2 = get_pred_pts3d(gt=None, pred=pred2, use_pose=True)
            gt_pts1 = get_pred_pts3d(gt=None, pred=gt1, use_pose=False)
            gt_pts2 = get_pred_pts3d(gt=None, pred=gt2, use_pose=True)

        else:
            valid1 = gt1['valid_mask'].clone()
            valid2 = gt2['valid_mask'].clone()

            # everything is normalized w.r.t. camera of view1
            in_camera1 = inv(gt1['camera_pose'])
            gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
            gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

            if dist_clip is not None:
                # points that are too far-away == invalid
                dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
                dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
                valid1 = valid1 & (dis1 <= dist_clip)
                valid2 = valid2 & (dis2 <= dist_clip)

            pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
            pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)

        
        # Normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and (not self.gt_scale):
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        # loss on img1 side
        l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
        # loss on gt2 side
        if self.asimmetric:
            l2 = l1
        else:
            l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])
        self_name = type(self).__name__
        details = {self_name+'_pts3d': float(l1.mean() + l2.mean())/2}
        # roma loss
        if self.roma:
            with torch.no_grad():
                renorm_img1 = (gt1['img'] * 0.5 + 0.5 - ROMA_MEAN.to(gt_pts1.device)) / ROMA_STD.to(gt_pts1.device)  
                renorm_img2 = (gt2['img'] * 0.5 + 0.5 - ROMA_MEAN.to(gt_pts1.device)) / ROMA_STD.to(gt_pts1.device)
                warp, certainty = self.roma.match(renorm_img1, renorm_img2, batched=True, device=gt_pts1.device)
            warp, certainty = warp.reshape(-1, 2*H*W, 4), certainty.reshape(-1, 2*H*W)
            kptsA, kptsB = self.roma.to_pixel_coordinates(warp, H, W, H, W)
            kptsA, kptsB = kptsA.type(torch.int64), kptsB.type(torch.int64)
            kptsA, kptsB = kptsA.reshape(-1,H,2*W,2), kptsB.reshape(-1,H,2*W,2)

            kpts1 = kptsA[:,:,:W,:] # B, H, W, 2
            kpts2 = kptsB[:,:,:W,:] # B, H, W, 2
            conf = pred1['conf']
            pred1 = pred1['pts3d'] # -> kpts1
            pred2 = pred2['pts3d_in_other_view'] # -> kpts2
            kpts1, kpts2 = kpts1.reshape(-1,H*W,2), kpts2.reshape(-1,H*W,2)
            kpts1_flat = torch.from_numpy(np.ravel_multi_index(kpts1.cpu().permute(-1,0,1).numpy(), (H, W), order='F')).to(gt_pts1.device)
            kpts2_flat = torch.from_numpy(np.ravel_multi_index(kpts2.cpu().permute(-1,0,1).numpy(), (H, W), order='F')).to(gt_pts1.device)
            pred1_flat = pred1.reshape(-1, H*W, 3)
            pred2_flat = pred2.reshape(-1, H*W, 3)
            p1 = pred1_flat.gather(1, kpts1_flat.unsqueeze(-1).expand(-1,-1,3))
            p2 = pred2_flat.gather(1, kpts2_flat.unsqueeze(-1).expand(-1,-1,3))

            # weight
            # cert = certainty.reshape(-1,H,2*W)[:,:,:W].reshape(-1,H*W)
            # conf = conf.reshape(-1,H*W)
            # conf = torch.sigmoid(torch.log(conf))

            # mask
            cert = (certainty.reshape(-1,H,2*W)[:,:,:W].reshape(-1,H*W) > self.roma_thr).float() # 
            conf = (conf > self.conf_thr).reshape(-1,H*W).float() # .reshape(-1,H*W) ###################################################

            # overlap
            # cert > 0.5 must have more than 1% of the pixels to have enough overlap
            # cert = (certainty.reshape(-1,H,2*W)[:,:,:W].reshape(-1,H*W) > self.roma_thr).float() # 
            # conf = torch.ones_like(cert)

            m = cert * conf
            p1c = p1 * m.unsqueeze(-1)
            p2c = p2 * m.unsqueeze(-1)
            
            # if m.mean() < 0.01:
            #     rl1, rl2 = torch.tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)

            # frame_diff = torch.abs(gt1['index'] - gt2['index'])
            # mask = frame_diff < 5
            # if not m.any():
            #     rl1, rl2 = torch.tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
            # else:
            #     p1c, p2c = p1c[mask], p2c[mask]
            #     rl2 = ((p1c - p2c)**2).mean() / cert.mean() / mask.float().mean()
            #     rl1 = (p1c - p2c).abs().mean() / cert.mean() / mask.float().mean()
            # else:
            rl2 = ((p1c - p2c)**2).mean() / (m > 0).float().mean()
            rl1 = (p1c - p2c).abs().mean() / (m > 0).float().mean()

            # cert = (certainty.reshape(-1,H,2*W)[:,:,:W].reshape(-1,H*W) > self.roma_thr).float() # 
            # conf = (conf > 2).reshape(-1,H*W).float() # .reshape(-1,H*W)
            # m = cert * conf
            # p1c = p1 * m.unsqueeze(-1)
            # p2c = p2 * m.unsqueeze(-1)

            # frame_diff = torch.abs(gt1['index'] - gt2['index'])
            # mask = frame_diff < 5
            # if not m.any():
            #     rl1, rl2 = torch.tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
            # else:
            #     p1c, p2c = p1c[mask], p2c[mask]
            #     rl2 = ((p1c - p2c)**2).mean() / cert.mean() / mask.float().mean()
            #     rl1 = (p1c - p2c).abs().mean() / cert.mean() / mask.float().mean()
            # else:
            #     rl2 = ((p1c - p2c)**2).mean() / m.mean()
            #     rl1 = (p1c - p2c).abs().mean() / m.mean()
            
            details['roma_mse'] = rl2
            details['roma_mae'] = rl1
        
        if self.debug:
            details['roma_details'] = (p1.detach().cpu(), p2.detach().cpu(), 
                                       certainty.reshape(-1,H,2*W)[:,:,:W].reshape(-1,H*W).detach().cpu())
            details['scaled_outs'] = (gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2)

        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        return conf_loss1 + conf_loss2, dict(conf_loss=float(conf_loss1 + conf_loss2)/2, **details)


class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, kd=False):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2, kd=kd)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, kd=False):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2, kd=kd                          )

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass
