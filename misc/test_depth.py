from data.megadepth import CreateDataLoader
import torchvision.transforms as tvf
import numpy as np
ImgNorm = tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference, load_model
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from tqdm import tqdm
from contextlib import contextmanager
import sys, os, copy
import torch
from dust3r.model import AsymmetricCroCo3DStereo, inf

dataset_root = '/ssd1/sa58728/dust3r/data/'
list_dir = '/ssd1/sa58728/dust3r/data/MegaDepth_v1/test_list/landscape/'
input_height = 224
input_width = 224
is_flipped = False
shuffle = False

teacher_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
student_path = 'log/train_10_1%_r/checkpoint-best.pth'

device = 'cuda:7'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

KD = True

def main():
    data_loader_l = CreateDataLoader(dataset_root, list_dir, input_height, input_width, is_flipped, shuffle)
    dataset_l = data_loader_l.load_data()
    dataset_size_l = len(data_loader_l)
    print('test images = %d' % dataset_size_l)

    teacher = load_model(teacher_path, device).eval()

    if KD:
        model = copy.deepcopy(teacher)
        model.to(device)

        model_kd = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
                    output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                    enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, adapter=True)"

        model_kd = eval(model_kd)
        model_kd.to(device)
        ckpt = torch.load(student_path, map_location=device)
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
    else:
        model = teacher

    mse, mae, rae, inliers, n = 0, 0, 0, 0, 0
    for i, data in tqdm(enumerate(dataset_l)):
        img = data['img_1']
        mask = data['target_1']['mask_0']  
        gt = data['target_1']['gt_0']

        for j in range(len(img)-1):
            scaled_gt = gt[j] / gt[j].max()
            gt_mask = scaled_gt >= 1e-2
            masked_gt = scaled_gt * gt_mask

            with suppress_stdout():
                imgs = prep_images(img[j], img[j+1])
                pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=False)
                output = inference(pairs, model, device, batch_size=1)
                scene = global_aligner(output[0], device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
                _ = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

            depths = to_numpy(scene.get_depthmaps())
            depths_max = max([d.max() for d in depths])
            depths = [d/depths_max for d in depths]

            test = depths[0] * gt_mask.numpy()
            test = test / test.max()

            mse += np.mean((test-masked_gt.numpy())**2)
            mae += np.mean(np.abs((test-masked_gt.numpy())))
            rae += m_rel_ae(masked_gt.numpy(), test, mask=mask[j].numpy())
            inliers += thresh_inliers(masked_gt.numpy(), test, 1.03, mask=mask[j].numpy())
            n += 1
        
        if i == 1:
            break
    
    mse /= n
    mae /= n
    rae /= n
    inliers /= n
    print(f"MSE: {mse}, MAE: {mae}, RAE: {rae}, Inliers: {inliers}")

def prep_images(image1, image2):
    imgs = []
    for img in [image1, image2]:
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.array([[224,224]]), idx=len(imgs), instance=str(len(imgs))))

    return imgs


def thresh_inliers(gt, pred, thresh, mask=None, output_scaling_factor=1.0):
    """Computes the inlier (=error within a threshold) ratio for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        thresh: Threshold for the relative difference between the prediction and ground truth.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Scalar that indicates the inlier ratio. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_1 = np.nan_to_num(gt / pred, nan=thresh+1, posinf=thresh+1, neginf=thresh+1)  # pred=0 should be an outlier
        rel_2 = np.nan_to_num(pred / gt, nan=0, posinf=0, neginf=0)  # gt=0 is masked out anyways

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(np.float32)  # 1 for inliers, 0 for outliers

    inlier_ratio, valid = valid_mean(inliers, mask)

    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan

    return inlier_ratio


def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the mean-relative-absolute-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).


    Returns:
        Scalar that indicates the mean-relative-absolute-error. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    ae = np.abs(e)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ae = np.nan_to_num(ae / gt, nan=0, posinf=0, neginf=0)

    m_rel_ae, valid = valid_mean(rel_ae, mask)

    m_rel_ae = m_rel_ae * output_scaling_factor
    m_rel_ae = m_rel_ae if valid else np.nan

    return m_rel_ae


def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    """Compute mean of elements across given dimensions of an array, considering only valid elements.

    Args:
        arr: The array to compute the mean.
        mask: Array with numerical or boolean values for element weights or validity. For bool, False means invalid.
        axis: Dimensions to reduce.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
        Mean array/scalar and a valid array/scalar that indicates where the mean could be computed successfully.
    """

    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mean = masked_arr_sum / num_valid
        is_valid = np.isfinite(valid_mean)
        valid_mean = np.nan_to_num(valid_mean, copy=False, nan=0, posinf=0, neginf=0)

    return valid_mean, is_valid


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

if __name__ == '__main__':
    main()