# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Co3d_v2
# dataset at https://github.com/facebookresearch/co3d - Creative Commons Attribution-NonCommercial 4.0 International
# See datasets_preprocess/preprocess_co3d.py
# --------------------------------------------------------
import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np
import torch

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Co3d(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, features=False, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.features = features

        # load all scenes
        with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
            self.scenes = json.load(f)
            self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
            self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                           for k2, v2 in v.items()}
        self.scene_list = list(self.scenes.keys())

        if self.gaussian_frames:
            self.combinations = [(i, j)
                                for i, j in itertools.combinations(range(100), 2)
                                if abs(i-j) == 3
                                ]
        else:
            # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
            # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
            self.combinations = [(i, j)
                                for i, j in itertools.combinations(range(100), 2)
                                if 0 < abs(i-j) <= 30 and abs(i-j) % 5 == 0
                                ]

        self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = self.scenes[obj, instance]
        im1_idx, im2_idx = self.combinations[idx % len(self.combinations)]

        # add a bit of randomness
        last = len(image_pool)-1

        if resolution not in self.invalidate[obj, instance]:  # flag invalid images
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2)) # 50% chance

        views = []
        if self.gaussian_frames:
            imgs_idxs = [max(0, min(im_idx + int(rng.normal(loc=0.0, scale=3)), last)) for im_idx in [im2_idx, im1_idx]]
        else:
            imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            try:
                if self.invalidate[obj, instance][resolution][im_idx]:
                    # search for a valid image
                    random_direction = 2 * rng.choice(2) - 1
                    for offset in range(1, len(image_pool)):
                        tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                        if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                            im_idx = tentative_im_idx
                            break
            except:
                print(obj, instance, resolution, im_idx)
                raise ValueError
            view_idx = image_pool[im_idx]

            impath = osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.jpg')

            # load camera params
            input_metadata = np.load(impath.replace('jpg', 'npz'))
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(impath.replace('images', 'depths') + '.geometric.png', cv2.IMREAD_UNCHANGED)
            depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])

            if mask_bg:
                # load object mask
                maskpath = osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            num_valid = (depthmap > 0.0).sum()
            # if num_valid == 0:
            #     # problem, invalidate image and retry
            #     # print(f"Invalid image {impath} for {obj} {instance} {resolution} {im_idx}")
            #     self.invalidate[obj, instance][resolution][im_idx] = True
            #     imgs_idxs.append(im_idx)
            #     continue

            # load features
            if self.features:
                features = np.load(impath.replace('jpg', 'npy'))
                points = 0
            else:
                features, points = 0, 0

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='Co3d_v2',
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1],
                img_path=impath,
                points=points,
                features=features
            ))
        return views
