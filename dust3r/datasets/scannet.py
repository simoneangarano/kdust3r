import sys
sys.path.append('/workspace')
import os.path as osp
import json
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from collections import deque
import numpy as np
import itertools
import cv2
from dust3r.utils.image import imread_cv2
import pandas as pd
import open3d as o3d
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

def save_pcd(points, colors, filename):
    # flatten the points and colors
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    # create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

class Scannet(BaseStereoViewDataset):
    def __init__(self, test=False, random=False, shuffle=False, mask_bg=True, num_views=2, intervals=[1,2,3], *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.num_views = num_views
        self.intervals = intervals
        self.shuffle = shuffle
        self.random = random
        self.test = test
        self.combinations = {}
        self.cumulative_counts = []
        
        # load all scenes
        with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
            self.scenes = json.load(f)
            # self.scenes = {k: sorted(v) for k, v in self.scenes.items() if len(v) > 0}
        
        self.scene_list = list(self.scenes.keys())
        
        self.invalidate = {scene: {} for scene in self.scene_list}
        self.prepare_scene_combinations()
        
    def __len__(self):
        return self.cumulative_counts[-1]
    
    def prepare_scene_combinations(self):
        total = 0
        for scene_id, images in self.scenes.items():
            scene_combinations = list(self.generate_combinations(len(images)))
            self.combinations[scene_id] = scene_combinations
            count = len(scene_combinations)
            total += count
            self.cumulative_counts.append(total)
            
    def generate_combinations(self, num_images):
        n = num_images
        k = self.num_views - 1
        all_combinations = []

        for interval in self.intervals:
            for start in range(n):
                end = start + interval * k
                if end < n:
                    all_combinations.append((start, end))

        return all_combinations
                
    def get_image_pair(self, idx):
        for i, cumulative_count in enumerate(self.cumulative_counts):
            if idx < cumulative_count:
                local_idx = idx - (self.cumulative_counts[i - 1] if i > 0 else 0)
                scene_id = self.scene_list[i]
                break
        return scene_id, local_idx
    
    def _get_views(self, idx, resolution, rng):
        # choose a scene
        scene_id, local_idx = self.get_image_pair(idx)
        
        image_pool = self.scenes[scene_id]
        im1_idx, im2_idx = self.combinations[scene_id][local_idx]
        
        # add a bit of randomness
        last = len(image_pool)-1
        
        if resolution not in self.invalidate[scene_id]:  # flag invalid images
            self.invalidate[scene_id][resolution] = [False for _ in range(len(image_pool))]
            
        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))
        
        views = []
        interval = (im2_idx - im1_idx) // (self.num_views - 1)
        if self.random:
            imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in range(im1_idx, im2_idx + 1, interval)]
            imgs_idxs = sorted(imgs_idxs, reverse=True)
        else:
            imgs_idxs = [im_idx for im_idx in range(im1_idx, im2_idx + 1, interval)][::-1]
            
        if self.shuffle:
            rng.shuffle(imgs_idxs)
            
        imgs_idxs = deque(imgs_idxs)
        
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()
            
            if self.invalidate[scene_id][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[scene_id][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break
        
            view_idx = image_pool[im_idx]

            impath = osp.join(self.ROOT, scene_id, 'images', f'{view_idx}.jpg')
            meta_data_path = impath.replace('jpg', 'npz')
            depthmap_path = impath.replace('images', 'depths').replace('.jpg', '.png')
            if self.test:
                labelmap_path = impath.replace('images', 'labels').replace('.jpg', '.png')
            # maskpath = impath.replace('images', 'masks').replace('.jpg', '.png')
            
            # load camera params
            input_metadata = np.load(meta_data_path)
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            has_inf = np.isinf(camera_pose)
            contains_inf = np.any(has_inf)
            if contains_inf:
                self.invalidate[scene_id][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue
            
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
            
            # load image and depth and mask
            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED)
            maskmap = np.ones_like(depthmap) * 255 # don't use mask for now
            if self.test:
                labelmap = imread_cv2(labelmap_path, cv2.IMREAD_UNCHANGED)
                # pack
                depth_mask_map = np.stack([depthmap, maskmap, labelmap], axis=-1)
            else:
                depth_mask_map = np.stack([depthmap, maskmap], axis=-1)
                
            # crop if necessary
            rgb_image, depth_mask_map, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depth_mask_map, intrinsics, resolution, rng=rng, info=impath)
            # unpack
            depthmap = depth_mask_map[:, :, 0]
            maskmap = depth_mask_map[:, :, 1]
            
            depthmap = (depthmap.astype(np.float32) / 1000)
            if mask_bg:
                # load object mask
                maskmap = maskmap.astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[scene_id][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue
            view = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='Scannet',
                label=scene_id,
                instance=osp.split(impath)[1],
            )
            if self.test:
                view['labelmap'] = labelmap
            views.append(view)
            
        return views
    
    def get_test_views(self, scene_id, view_idx, resolution):
        if type(resolution) == int:
            resolution = (resolution, resolution)
        else:
            resolution = tuple(resolution)
            
        impath = osp.join(self.ROOT, scene_id, 'images', f'{view_idx}.jpg')
        meta_data_path = impath.replace('jpg', 'npz')
        depthmap_path = impath.replace('images', 'depths').replace('.jpg', '.png')
        labelmap_path = impath.replace('images', 'labels').replace('.jpg', '.png')
        
        # load camera params
        input_metadata = np.load(meta_data_path)
        camera_pose = input_metadata['camera_pose'].astype(np.float32)
        intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
        
        # load image and depth and mask
        rgb_image = imread_cv2(impath)
        depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED)
        maskmap = np.ones_like(depthmap) * 255 # don't use mask for now
        labelmap = imread_cv2(labelmap_path, cv2.IMREAD_UNCHANGED)
        
        # pack
        depth_mask_map = np.stack([depthmap, maskmap, labelmap], axis=-1)
        
        # crop if necessary
        rgb_image, depth_mask_map, intrinsics = self._crop_resize_if_necessary(
            rgb_image, depth_mask_map, intrinsics, resolution, rng=None, info=impath)
        
        # unpack
        depthmap = depth_mask_map[:, :, 0]
        maskmap = depth_mask_map[:, :, 1]
        labelmap = depth_mask_map[:, :, 2]
        
        depthmap = (depthmap.astype(np.float32) / 1000)
        # load object mask
        maskmap = maskmap.astype(np.float32)
        maskmap = (maskmap / 255.0) > 0.1

        # update the depthmap with mask
        depthmap *= maskmap
        
        view = dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose,
            labelmap=labelmap,
            camera_intrinsics=intrinsics,
            dataset='Scannet',
            label=scene_id,
            instance=osp.split(impath)[1],
        )
        assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
        view['idx'] = (view_idx)

        # encode the image
        width, height = view['img'].size
        view['true_shape'] = np.int32((height, width))
        view['img'] = self.transform(view['img'])

        assert 'camera_intrinsics' in view
        if 'camera_pose' not in view:
            view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
        else:
            assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
        assert 'pts3d' not in view
        assert 'valid_mask' not in view
        assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

        view['pts3d'] = pts3d
        view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
        
        return view
    
if __name__ == "__main__":
    import sys
    sys.path.append('/workspace')
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = Scannet(split='test', ROOT="data/scannet_processed", resolution=(512, 384))
    print(len(dataset))
    test_view = dataset.get_test_views('scene0000_00', 0, (512, 384))
    # for idx in np.random.permutation(len(dataset)):
    #     views = dataset[idx]
    #     # assert len(views) == 2
    #     print(view_name(views[0]), view_name(views[-1]))
    #     viz = SceneViz()
    #     poses = [views[view_idx]['camera_pose'] for view_idx in [0, -1]]
    #     cam_size = max(auto_cam_size(poses), 0.001)
    #     for view_idx in [0, 1]:
    #         img = views[view_idx]['img']
    #         pts3d = views[view_idx]['pts3d']
    #         # save pts3d to file
    #         pts3d_path = f'{view_idx}_scannetpp_pts3d.ply'
    #         # save_pcd(pts3d, img.permute(1, 2, 0).numpy(), pts3d_path)
    #         valid_mask = views[view_idx]['valid_mask']
    #         colors = rgb(views[view_idx]['img'])
    #         viz.add_pointcloud(pts3d, colors, valid_mask)
    #         viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
    #                        focal=views[view_idx]['camera_intrinsics'][0, 0],
    #                        color=(idx*255, (1 - idx)*255, 0),
    #                        image=colors,
    #                        cam_size=cam_size)
    #     viz.show()