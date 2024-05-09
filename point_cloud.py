from copy import deepcopy
import numpy as np
import torch
import open3d as o3d
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.model import AsymmetricCroCo3DStereo, inf

ENCODER_ONLY = True # if True, use teacher's decoder and student's encoder
CKPT = f'log/train_v2_2/checkpoint-best.pth' # student's checkpoint
TEACHER_CKPT = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" # teacher's checkpoint
TEST_SUBSETS = ['co3d_test_1', 'co3d_test_2', 'croco', 'dtu'] # image pairs to test

def main():

    device = 'cpu'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    if ENCODER_ONLY:
        model_dims = [384, 6, 768, 12]
    else:
        model_dims = [384, 6, 192, 3]
    MODEL_KD = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
                output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                enc_embed_dim={}, enc_depth=12, enc_num_heads={}, dec_embed_dim={}, dec_depth=12, dec_num_heads={}, adapter=True)".format(*model_dims)
    teacher, model = build_model_enc_dec(MODEL_KD, device)


    for s in TEST_SUBSETS:
        img_folder_path = f'test/{s}/'

        # load_images can take a list of images or a directory
        images = load_images(img_folder_path, size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

        result = inference(pairs, model, device, batch_size=batch_size, features_only=True)
        teacher_result = inference(pairs, teacher, device, batch_size=batch_size, features_only=True)

        loss, metric = 0, 0
        for pred_side, teacher_side in zip(result[0], teacher_result[0]): # for each view
            pred_feats, _ = pred_side
            target_feats, _ = teacher_side
            loss += mseloss(pred_feats, target_feats) / 2.0
            metric += maeloss(pred_feats, target_feats) / 2.0
        print(f"Loss: {loss.item()}, Metric: {metric.item()}")

        print("Teacher Scene")
        output = inference(pairs, teacher, device, batch_size=batch_size)
        scene = global_aligner(output[0], device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
        # retrieve useful values from scene:
        imgs = scene.imgs
        # focals = scene.get_focals()
        # poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        # confidence_masks = scene.get_masks()
        pts3d = to_numpy(pts3d)
        pts_4_3dgs = np.concatenate([item.reshape(-1, 3) for item in pts3d])
        color_4_3dgs = np.concatenate([item.reshape(-1, 3) for item in imgs])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_4_3dgs)
        pcd.colors = o3d.utility.Vector3dVector(color_4_3dgs)
        o3d.io.write_point_cloud(f"{img_folder_path}{s}.ply", pcd)

        print("Student Scene")
        output_kd = inference(pairs, model, device, batch_size=batch_size)
        scene = global_aligner(output_kd[0], device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
        # retrieve useful values from scene:
        imgs = scene.imgs
        # focals = scene.get_focals()
        # poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        # confidence_masks = scene.get_masks()
        pts3d = to_numpy(pts3d)
        pts_4_3dgs = np.concatenate([item.reshape(-1, 3) for item in pts3d])
        color_4_3dgs = np.concatenate([item.reshape(-1, 3) for item in imgs])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_4_3dgs)
        pcd.colors = o3d.utility.Vector3dVector(color_4_3dgs)
        o3d.io.write_point_cloud(f"{img_folder_path}{s}_kd.ply", pcd)


def mseloss(pred_feats, target_feats):
    return ((pred_feats - target_feats)**2).mean()
    # return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()

def maeloss(pred_feats, target_feats):
    return (torch.abs(pred_feats - target_feats)).mean()
    # return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()

def build_model_enc_dec(model_str, device):

    teacher = load_model(TEACHER_CKPT, device)
    teacher.eval()
    
    model = deepcopy(teacher)
    model.to(device)
    model.eval()

    model_kd = eval(model_str)
    model_kd.to(device)
    model_kd.eval()

    ckpt = torch.load(CKPT)
    model_kd.load_state_dict(ckpt['model'], strict=True)
    model_kd.train()

    if ENCODER_ONLY:
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
    main()