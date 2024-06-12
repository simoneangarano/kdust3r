import argparse
import random
import numpy as np
from test_kd import get_args_parser, build_dataset, build_model_enc_dec 

args = get_args_parser()
args = args.parse_args('')

H, W = 224, 224
TEST_DATA = f"BlendedMVS(split='train', ROOT='/ssd1/sa58728/dust3r/data/blendedmvs_processed', resolution=224, seed=777, gauss_std={args.gauss_std})"

args = get_args_parser()
args = args.parse_args('')

# SETUP
seed = args.seed
np.random.seed(seed)
random.seed(seed)

def tensor_to_plot(tensor, norm=None):
    STD = np.array([0.229, 0.224, 0.225])
    MEAN = np.array([0.485, 0.456, 0.406])
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if len(tensor.shape) == 2:
        return tensor.cpu().numpy()
    elif norm == 'roma':
        return tensor.cpu().permute(1,2,0).numpy() * STD + MEAN
    elif norm == 'dust3r':
        return (tensor.cpu().permute(1,2,0).numpy() * 0.5) + 0.5
    return tensor.cpu().permute(1,2,0).numpy()

# DATA
# TEST_DATA =  f"Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', resolution=224, seed=777, gauss_std={args.gauss_std})"
# TEST_DATA += f"+ ScanNet(split='test', ROOT='/ssd1/wenyan/scannetpp_processed', resolution=224, seed=777, gauss_std={args.gauss_std})"
# TEST_DATA += f"+ DL3DV(split='test', ROOT='/ssd1/sa58728/dust3r/data/DL3DV-10K', resolution=224, seed=777, gauss_std={args.gauss_std})"
TEST_DATA = f"BlendedMVS(split='train', ROOT='/ssd1/sa58728/dust3r/data/blendedmvs_processed', resolution=224, seed=777, gauss_std={args.gauss_std})"

data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                    for dataset in TEST_DATA.split('+')}

for test_name, testset in data_loader_test.items():
    print(test_name)
    if hasattr(testset, 'dataset') and hasattr(testset.dataset, 'set_epoch'):
        testset.dataset.set_epoch(0)
    if hasattr(testset, 'sampler') and hasattr(testset.sampler, 'set_epoch'):
        testset.sampler.set_epoch(0)

    for _, batch in enumerate(testset):
        view1, view2 = batch
        
        img = tensor_to_plot(view1['img'][0], norm='dust3r')
        pts = tensor_to_plot(view1['pts3d'][0,...,-1].detach())
        