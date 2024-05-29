#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the CO3D dataset.
# Usage:
# python3 datasets_preprocess/preprocess_co3d.py --co3d_dir /path/to/co3d
# --------------------------------------------------------

import argparse
import random
import gzip
import json
import os
import os.path as osp
import re
import torch
import PIL.Image
import numpy as np
import cv2

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import data.path_to_root  # noqa
import dust3r.datasets.utils.cropping as cropping  # noqa

VIEWS = 1
MAX_D = 128
SAMPLE_SCALE = 0.25
SCALE_FACTOR = 1.0
DOWNSAMPLE = 1.0

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/ssd1/sa58728/dust3r/data/blendedmvs_processed")
    parser.add_argument("--dtu_dir", type=str, default="/ssd1/sa58728/dust3r/data/BlendedMVS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=512,
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--overwrite", default=True)
    return parser


def gen_blendedmvs_path(blendedmvs_data_folder, mode='train'):
    """ generate data paths for blendedmvs dataset """
    # read data list
    if mode == 'train':
        proj_list = open(os.path.join(blendedmvs_data_folder, 'BlendedMVS_training.txt')).read().splitlines()
    elif mode == 'val':
        proj_list = open(os.path.join(blendedmvs_data_folder, 'validation_list.txt')).read().splitlines()

    # parse all data
    mvs_input_list = []
    for data_name in proj_list:

        dataset_folder = os.path.join(blendedmvs_data_folder, data_name)

        # read cluster
        cluster_path = os.path.join(dataset_folder, 'cams', 'pair.txt')
        cluster_lines = open(cluster_path).read().splitlines()
        image_num = int(cluster_lines[0])

        # get per-image info
        for idx in range(0, image_num):

            ref_idx = int(cluster_lines[2 * idx + 1])
            cluster_info =  cluster_lines[2 * idx + 2].split()
            total_view_num = int(cluster_info[0])
            if total_view_num < VIEWS - 1:
                continue
            paths = {}
            ref_image_path = os.path.join(dataset_folder, 'blended_images', '%08d_masked.jpg' % ref_idx)
            ref_depth_path = os.path.join(dataset_folder, 'rendered_depth_maps', '%08d.pfm' % ref_idx)
            ref_cam_path = os.path.join(dataset_folder, 'cams', '%08d_cam.txt' % ref_idx)
            paths['image'] = ref_image_path
            paths['cam'] = ref_cam_path
            paths['depth'] = ref_depth_path
            mvs_input_list.append([data_name, ref_idx, paths])

    return mvs_input_list


def load_cam(file, interval_scale=1, max_d=128):
    """ read camera txt file """
    words = file.read().split()

    # read extrinsic
    extr = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            extr[i][j] = words[extrinsic_index]

    # read intrinsic
    intr = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            intr[i][j] = words[intrinsic_index]

    info = np.zeros((4))
    if len(words) == 29:
        info[0] = words[27]
        info[1] = float(words[28]) * interval_scale
        info[2] = max_d
        info[3] = info[0] + info[1] * info[2]
    elif len(words) == 30:
        info[0] = words[27]
        info[1] = float(words[28]) * interval_scale
        info[2] = words[29]
        info[3] = info[0] + info[1] * info[2]
    elif len(words) == 31:
        info[0] = words[27]
        info[1] = float(words[28]) * interval_scale
        info[2] = words[29]
        info[3] = words[30]

    return extr, intr, info

def load_cam_old(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = MAX_D
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image

def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam

def scale_mvs_camera(cams, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(VIEWS):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) * SCALE_FACTOR
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * SCALE_FACTOR
    depth_interval = float(lines[11].split()[1])
    return intrinsics, extrinsics, [depth_min, depth_max], depth_interval

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def _read_pfm_disp(filename):
    disp = np.ascontiguousarray(_read_pfm(filename)[0])
    disp[disp<=0] = np.inf # eg /nfs/data/ffs-3d/datasets/middlebury/2014/Shopvac-imperfect/disp0.pfm
    return disp


def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def read_depth(filename):
    depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
    # depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
    #                         interpolation=cv2.INTER_NEAREST)  # (600, 800)
    # depth_h = depth_h[44:556, 80:720]  # (512, 640)
    depth_h = cv2.resize(depth_h, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE,
                            interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
    depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
                        interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
    mask = depth > 0
    mask_h = depth_h > 0

    return depth, mask, depth_h, mask_h
    
def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def prepare_sequences(bmvs_dir, output_dir, img_size, split, subset, seed):
    random.seed(seed)

    sequences_all = gen_blendedmvs_path(bmvs_dir, mode=split)
    selected_sequences_numbers_dict = {}
    for j, (seq_name, frame_number, filepaths) in tqdm(enumerate(sequences_all)):
        if seq_name not in selected_sequences_numbers_dict:
            selected_sequences_numbers_dict[seq_name] = []
        image_path = filepaths['image']
        try:
            input_rgb_image = PIL.Image.open(image_path).convert('RGB')
        except:
            continue
        selected_sequences_numbers_dict[seq_name].append(frame_number)

        # camera_extrinsics, camera_intrinsics, camera_info = load_cam(open(filepaths['cam'])) 
        # cams = [load_cam_old(open(filepaths['cam']))]
        intrinsic, extrinsic, near_far, depth_interval = read_cam_file(filepaths['cam'])
        intrinsic[:2] *= 4
        extrinsic[:3, 3] *= SCALE_FACTOR
        intrinsic[:2] = intrinsic[:2] * DOWNSAMPLE
        # multiply intrinsics and extrinsics to get projection matrix
        proj_mat_l = np.eye(4)
        intrinsic[:2] = intrinsic[:2] / 4
        proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]
        camera_pose = np.linalg.inv(proj_mat_l)
        camera_intrinsics = intrinsic

        depth_path = filepaths['depth']
        _, _, depth_h, input_mask = read_depth(depth_path)
        depth_h *= SCALE_FACTOR
        input_depthmap = depth_h
        # input_depthmap = np.nan_to_num(_read_pfm_disp(depth_path), posinf=0, nan=0, neginf=0)

        # downsize by 4 to fit depth map output
        # input_depthmap = scale_image(input_depthmap, scale=SAMPLE_SCALE)
        # cams = scale_mvs_camera(cams, scale=SAMPLE_SCALE)
        # fix depth range and adapt depth sample number 
        # cams[0][1, 3, 2] = MAX_D
        # cams[0][1, 3, 1] = (cams[0][1, 3, 3] - cams[0][1, 3, 0]) / MAX_D
        # # mask out-of-range depth pixels (in a relaxed range)
        # depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
        # depth_end = cams[0][1, 3, 0] + (MAX_D - 2) * cams[0][1, 3, 1]
        # input_depthmap = mask_depth_image(input_depthmap, depth_start, depth_end).squeeze()

        # input_mask = input_depthmap > 0.0
        input_depthmap *= input_mask
        depth_mask = np.stack((input_depthmap, input_mask), axis=-1)
        H, W = input_depthmap.shape

        # camera_extrinsics = cams[0][0]
        # camera_intrinsics = cams[0][1][:3, :3]

        cx, cy = camera_intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)

        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        input_rgb_image, depth_mask, input_camera_intrinsics = cropping.crop_image_depthmap(
            input_rgb_image, depth_mask, camera_intrinsics, crop_bbox)

        # try to set the lower dimension to img_size * 3/4 -> img_size=512 => 384
        scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
        output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
        if max(output_resolution) < img_size:
            # let's put the max dimension to img_size
            scale_final = (img_size / max(H, W)) + 1e-8
            output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

        input_rgb_image, depth_mask, input_camera_intrinsics = cropping.rescale_image_depthmap(
            input_rgb_image, depth_mask, input_camera_intrinsics, output_resolution)
        input_depthmap = depth_mask[:, :, 0]
        input_mask = depth_mask[:, :, 1]

        # generate and adjust camera pose
        # camera_pose = camera_extrinsics
        # camera_pose = np.linalg.inv(camera_pose)

        # save crop images and depth, metadata
        save_img_path = os.path.join(output_dir, filepaths['image'].split('BlendedMVS/')[-1])
        save_depth_path = os.path.join(output_dir, filepaths['depth'].split('BlendedMVS/')[-1].replace('.pfm', '.png'))
        save_mask_path = os.path.join(output_dir, filepaths['depth'].split('BlendedMVS/')[-1].replace('.pfm', '_mask.png'))
        os.makedirs(os.path.split(save_img_path)[0], exist_ok=True)
        os.makedirs(os.path.split(save_depth_path)[0], exist_ok=True)
        os.makedirs(os.path.split(save_mask_path)[0], exist_ok=True)

        input_rgb_image.save(save_img_path)
        scaled_depth_map = (input_depthmap / np.max(input_depthmap) * 65535).astype(np.uint16)
        cv2.imwrite(save_depth_path, scaled_depth_map)
        cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))

        save_meta_path = save_img_path.replace('jpg', 'npz')
        np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                 camera_pose=camera_pose, maximum_depth=np.max(input_depthmap))
        
        if subset is not None and j >= subset:
            break

    return selected_sequences_numbers_dict



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert args.dtu_dir != args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    for split in ['train', 'val']:
        selected_sequences_path = os.path.join(args.output_dir, f'selected_seqs_{split}.json')
        if os.path.isfile(selected_sequences_path) and not args.overwrite:
            continue

        print(f"Processing {split}")
        selected_sequences = prepare_sequences(
            bmvs_dir=args.dtu_dir,
            output_dir=args.output_dir,
            img_size=args.img_size,
            split=split,
            subset=args.subset,
            seed=args.seed,
        )

        with open(selected_sequences_path, 'w') as file:
            json.dump(selected_sequences, file)