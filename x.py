from dust3r.datasets import MegaDepth

dataset = MegaDepth(split='train', ROOT='/ssd1/sa58728/dust3r/data/MegaDepth_v1', aug_crop=16, mask_bg='rand', resolution=224)

for i in dataset:
    print(i[0]['img'].shape)