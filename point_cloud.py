from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.model import AsymmetricCroCo3DStereo, inf
import numpy as np
import torch
import open3d as o3d
import copy

def main():
    lmd = "10" 
    model_path = "checkpoints/DUSt3R_ViTSmall_BaseDecoder_512_dpt_kd.pth"
    model_kd_path = f'log/train_2/checkpoint-best.pth' # "checkpoints/DUSt3R_ViTSmall_BaseDecoder_512_dpt_kd.pth"
    model_kd = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
                output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, adapter=True)"
            # "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
            # output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
            # enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)"
    teacher_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    
    test_subsets = ['co3d_test_1', 'co3d_test_2', 'croco', 'dtu'] # ['croco', 'dtu', 'co3d_test_1', 'co3d_test_2']

    device = 'cuda:7'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    teacher = load_model(teacher_path, device)
    teacher.eval()

    # model = eval(model_kd)
    model = copy.deepcopy(teacher)
    model.to(device)
    # ckpt = torch.load(model_path, map_location=device)
    # try:
    #     print(model.load_state_dict(ckpt, strict=True))
    # except:
    #     print(model.load_state_dict(ckpt['model'], strict=True))
    # del ckpt  # in case it occupies memory
    # model.eval()

    model_kd = eval(model_kd)
    model_kd.to(device)
    ckpt = torch.load(model_kd_path, map_location=device)
    try:
        print(model_kd.load_state_dict(ckpt, strict=True))
    except:
        print(model_kd.load_state_dict(ckpt['model'], strict=True))
    del ckpt  # in case it occupies memory

    model.patch_embed = copy.deepcopy(model_kd.patch_embed)
    model.mask_generator = copy.deepcopy(model_kd.mask_generator)
    model.rope = copy.deepcopy(model_kd.rope)
    model.enc_blocks = copy.deepcopy(model_kd.enc_blocks)
    model.enc_norm = copy.deepcopy(model_kd.enc_norm)
    model.adapter = copy.deepcopy(model_kd.adapter)

    # model = load_model(model_path, device)

    # teacher, model = build_model_enc_dec(model_kd, model_kd_path, device)

    for s in test_subsets:
        img_folder_path = f'test/{s}/'

        # load_images can take a list of images or a directory
        images = load_images(img_folder_path, size=512)
        # img_A = os.path.join(img_folder_path, f"0.png")
        # img_B = os.path.join(img_folder_path, f"1.png")
        # images = load_images([img_A, img_B], size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

        result = inference(pairs, model, device, batch_size=batch_size, features_only=True)
        teacher_result = inference(pairs, teacher, device, batch_size=batch_size, features_only=True)

        loss, metric = 0, 0
        for pred_side, teacher_side in zip(result[0], teacher_result[0]): # for each view
            pred_feats, _ = pred_side
            target_feats, _ = teacher_side
            print(pred_feats.min(), pred_feats.max(), target_feats.min(), target_feats.max())
            loss += customized_mseloss(pred_feats, target_feats) / 2.0
            metric += customized_maeloss(pred_feats, target_feats) / 2.0

        print(f"Loss: {loss.item()}, Metric: {metric.item()}")

        print("Teacher")
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

        print("Student")
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


def customized_mseloss(pred_feats, target_feats):
    return ((pred_feats - target_feats)**2).mean()
    # return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()

def customized_maeloss(pred_feats, target_feats):
    return (torch.abs(pred_feats - target_feats)).mean()
    # return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()

def build_model_enc_dec(model_kd, model_kd_path, device):
    teacher = load_model("checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", device)
    teacher.eval()

    model = eval(model_kd)
    model.to(device)
    model.eval()
    ckpt = torch.load(model_kd_path, map_location=device)['model']
    model.load_state_dict(ckpt, strict=True)

    module_list = ['decoder_embed', 'dec_blocks', 'dec_norm', 'dec_blocks2', 'downstream_head1', 'downstream_head2']
    for m in module_list:
        getattr(model, m).load_state_dict(getattr(teacher, m).state_dict(), strict=True)
        getattr(model, m).eval()
    model.mask_token = teacher.mask_token

    return teacher, model

if __name__ == '__main__':
    main()