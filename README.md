`train_kd.py`: train KD model (check args in the file)
```console
python train_kd.py --ckpt log/train_2_v2/checkpoint-best.pth --encoder_only --gauss_std 3 --lmd 10 --batch_size 8 --output_dir logs/train/ --cuda 0
```

`test.py`: test KD model (check args in the file)
```console
python test.py --ckpt log/train_2_v2/checkpoint-best.pth --encoder_only --gauss_std 3 --batch_size 8 --cuda 0
```

`point_cloud.py`: generate teacher and student point clouds from image pairs (check parameters in the file)
```console
python point_cloud.py --ckpt log/train_2_v2/checkpoint-best.pth --encoder_only --test_pairs co3d_test_1 co3d_test_2 croco dtu --cuda 0
```
