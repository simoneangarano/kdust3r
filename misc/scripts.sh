CUDA_VISIBLE_DEVICES=1,3,4,5,7 torchrun --nproc_per_node=4 extract_features.py \
    --test_dataset "1000 @ Co3d(split='train', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777) + 100 @ Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --epochs 1 --batch_size 8 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 \
    --output_dir checkpoints/extracted/

# Encoder
# enc_embed_dim=1024 -> 192
# enc_depth=24 -> 12
# enc_num_heads=16 -> 3
# dec_embed_dim=768 
# dec_depth=12
# dec_num_heads=12

# ViT-Large
# enc_embed_dim=1024
# enc_depth=24
# enc_num_heads=16

# ViT-Small
# enc_embed_dim=384
# enc_depth=12
# enc_num_heads=6

# ViT-Tiny
# enc_embed_dim=192
# enc_depth=12
# enc_num_heads=3

# TinyViT
# custom

# Decoder

# ViT-Base
# dec_embed_dim=768
# dec_depth=12 
# dec_num_heads=12

# ViT-Small
# dec_embed_dim=384
# dec_depth=12
# dec_num_heads=6

# ViT-Tiny
# dec_embed_dim=192
# dec_depth=12
# dec_num_heads=3

# TinyViT
# custom