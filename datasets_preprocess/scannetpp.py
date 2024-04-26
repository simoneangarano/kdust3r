import argparse
import os
from pathlib import Path
from tqdm import tqdm
import os.path as osp
import json
import random

import sys
sys.path.append('scannetpp')
from common.scene_release import ScannetppScene_Release
from dslr.undistort import compute_undistort_intrinsic
from dslr.downscale import compute_resize_intrinsic
from common.utils.utils import load_json
from common.utils.colmap import read_model
import PIL.Image
import numpy as np
sys.path.append('dust3r')
import dust3r.datasets.utils.cropping as cropping  # noqa
import cv2
import imageio

scene_ids = ['ada5304e41', '210f741378', 'ed2216380b', '6cc2231b9c', '3e928dc2f6', 
             '394a542a19', 'b1d75ecd55', '8a20d62ac0', '8b2c0938d6', '7e7cd69a59', 
             'bc2fce1d81', '260fa55d50', '712dc47104', '2b1dc6d6a5', 'b5918e4637', 
             'bc03d88fc3', '25927bb04c', 'f25f5e6f63', '56a0ec536c', '5fb5d2dbf2', 
             'cbd4b3055e', '45d2e33be1', 'faec2f0468', '7977624358', 'e898c76c1f', 
             'acd69a1746', '0e75f3c4d9', '85251de7d1', '961911d451', 'a4e227f506', 
             '16c9bd2e1e', '303745abc7', 'e9e16b6043', 'f8f12e4e6b', '94ee15e8ba', 
             '4ba22fa7e4', '0cf2e9402d', '07f5b601ee', '08bbbdcc3d', '98fe276aa8', 
             '2a496183e1', '1a8e0d78c0', 'a1d9da703c', 'd6702c681d', 'e8e81396b6', 
             '37ea1c52f0', '69e5939669', '419cbe7c11', '4c5c60fa76', 'ebc200e928', 
             '108ec0b806', 'bf6e439e38', '87f6d7d564', '260db9cf5a', '41b00feddb', 
             'f07340dfea', '5a269ba6fe', '1ae9e5d2a6', 'e01b287af5', '07ff1c45bb', 
             'f34d532901', 'b20a261fdf', '116456116b', '1831b3823a', '1b75758486', 
             '1366d5ae89', '55b2bf8036', '88627b561e', '13285009a4', '30f4a2b44d', 
             'f8062cb7ce', '30966f4c6e', '1b9692f0c7', '8b5caf3398', '302a7f6b67', 
             '28a9ee4557', '9f139a318d', 'dfac5b38df', 'd6d9ddb03f', '47b37eb6f9', 
             'e3ecd49e2b', '49a82360aa', '52599ae063', 'bc400d86e1', 'a08dda47a8', 
             'c8f2218ee2', '50809ea0d8', '0a5c013435', 'c0c863b72d', '4a1a3a7dc5', 
             '2970e95b65', '646af5e14b', '9b74afd2d2', 'ad2d07fd11', 'c24f94007b', 
             'd415cc449b', '39f36da05b', '9471b8d485', '036bce3393', 'e050c15a8d', 
             'c9abde4c4b', '7cd2ac43b4']

scene_ids = scene_ids
data_root = '/mnt/vita-nas/scannetpp/'
downscale_factor = 3.0

SCENES_IDX = {scene_id: i for i, scene_id in enumerate(scene_ids)}  # for seeding

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/ssd1/sa58728/dust3r/data/scannetpp_processed")
    parser.add_argument("--scannetpp_dir", type=str, default="scannetpp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=512,
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    return parser
    

def prepare_sequences(scene_id, output_dir, img_size, split, seed):
    random.seed(seed)
    scene = ScannetppScene_Release(scene_id, data_root=Path(data_root) / "data")
    cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
    assert len(cameras) == 1, "Multiple cameras not supported"
    
    camera = next(iter(cameras.values()))
    fx, fy, cx, cy = camera.params[:4]
    params = camera.params[4:]
    # Undistort the camera intrinsics 
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1],])
    height = camera.height
    width = camera.width
    new_K = compute_undistort_intrinsic(K, height, width, params)
    # Resize the camera intrinsics
    scale_factor = 1 / downscale_factor
    new_K, new_height, new_width = compute_resize_intrinsic(new_K, height, width, scale_factor)
    new_fx, new_fy, new_cx, new_cy = new_K[0,0], new_K[1,1], new_K[0,2], new_K[1,2]

    new_camera_intrinsics = np.array([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]])
    images_list = load_json(scene.dslr_train_test_lists_path)[split]
    
    images_selected = []
    for image_id, image in tqdm(images.items()):
        image_name = image.name
        if image_name not in images_list:
            continue
        depth_name = image_name.split(".")[0] + ".png"
        
        image_idx = int(image.name[3:-4])
        images_selected.append(image_idx)
        
        image_path = scene.dslr_dir / 'undistorted_resized_images_3' / image_name
        mask_path = scene.dslr_dir / 'undistorted_resized_anon_masks_3' / depth_name
        depth_path = scene.dslr_dir / 'render_depth' / depth_name
        semseg_path = scene.dslr_dir / 'render_semantic' / depth_name
        
        input_rgb_image = PIL.Image.open(image_path).convert('RGB')
        input_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        input_depthmap = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        input_semsegmap = cv2.imread(str(semseg_path), cv2.IMREAD_UNCHANGED)
        
        H, W = new_height, new_width
        new_cx, new_cy = new_camera_intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(new_cx, W-new_cx)
        min_margin_y = min(new_cy, H-new_cy)
        
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = new_cx - min_margin_x, new_cy - min_margin_y
        r, b = new_cx + min_margin_x, new_cy + min_margin_y
        crop_bbox = (l, t, r, b)
        input_rgb_image, input_mask, input_depthmap, input_semsegmap, input_camera_intrinsics = cropping.crop_all(
            input_rgb_image, input_mask, input_depthmap, input_semsegmap, new_camera_intrinsics, crop_bbox)
        
        # try to set the lower dimension to img_size * 3/4 -> img_size=512 => 384
        scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
        output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
        if max(output_resolution) < img_size:
            # let's put the max dimension to img_size
            scale_final = (img_size / max(H, W)) + 1e-8
            output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

        input_rgb_image, input_mask, input_depthmap, input_semsegmap, input_camera_intrinsics = cropping.rescale_all(
            input_rgb_image, input_mask, input_depthmap, input_semsegmap, input_camera_intrinsics, output_resolution)
        
        camera_pose = np.linalg.inv(image.world_to_camera)
        
        # save crop images and depth, metadata
        save_img_path = os.path.join(output_dir, scene_id, 'images', image_name)
        save_depth_path = os.path.join(output_dir, scene_id, 'depths', depth_name)
        save_mask_path = os.path.join(output_dir, scene_id, 'masks', depth_name)
        save_semseg_path = os.path.join(output_dir, scene_id, 'semseg', depth_name)
        
        os.makedirs(os.path.split(save_img_path)[0], exist_ok=True)
        os.makedirs(os.path.split(save_depth_path)[0], exist_ok=True)
        os.makedirs(os.path.split(save_mask_path)[0], exist_ok=True)
        os.makedirs(os.path.split(save_semseg_path)[0], exist_ok=True)

        input_rgb_image.save(save_img_path)
        cv2.imwrite(save_depth_path, input_depthmap)
        cv2.imwrite(save_semseg_path, input_semsegmap)
        cv2.imwrite(save_mask_path, input_mask)
        save_meta_path = save_img_path.replace('JPG', 'npz')
        np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,camera_pose=camera_pose)
    return images_selected
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert args.scannetpp_dir != args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'test']:
        selected_sequences_path = os.path.join(args.output_dir, f'selected_seqs_{split}.json')
        if os.path.isfile(selected_sequences_path):
            continue
        all_selected_sequences = {}
        for scene_id in tqdm(scene_ids, desc="scene"):
            scene_output_dir = osp.join(args.output_dir, scene_id)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            scene_selected_sequences_path = os.path.join(scene_output_dir, f'selected_seqs_{split}.json')
            
            if os.path.isfile(scene_selected_sequences_path):
                with open(scene_selected_sequences_path, 'r') as fid:
                    scene_selected_sequences = json.load(fid)
            else:
                print(f"Processing {split} - scene = {scene_id}")
                scene_selected_sequences = prepare_sequences(
                    scene_id=scene_id,
                    output_dir=args.output_dir,
                    img_size=args.img_size,
                    split=split,
                    seed=args.seed + SCENES_IDX[scene_id],
                )
                with open(scene_selected_sequences_path, 'w') as file:
                    json.dump(scene_selected_sequences, file)
            all_selected_sequences[scene_id] = scene_selected_sequences
        with open(selected_sequences_path, 'w') as file:
            json.dump(all_selected_sequences, file)