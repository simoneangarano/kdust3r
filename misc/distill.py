import os
import numpy as np
import argparse
import random

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

from dust3r.datasets import get_data_loader  # noqa
from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.inference import load_model
from dust3r.utils import misc
# from tiny_vit import TinyViT
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

CKPT = None # "/home/sa58728/dust3r/log_lr/ckpt/iter_24750.pth"

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # dataset paths
    # parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b", help='root path of dataset')
    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=8, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    # multi gpu settings
    parser.add_argument("--local_rank", type=int, default=5)
    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=777, help='seed')
    parser.add_argument('--deterministic', type=bool, default=False, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')
    # learning process settings
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=100, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')
    # file and folder paths
    parser.add_argument('--root_path', type=str, default="./", help='root path')
    parser.add_argument('--teacher_path', type=str, default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help='teacher model path')
    parser.add_argument('--work_dir', type=str, default="log_kd/log_x", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpt", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=500, help='save iterations')
    parser.add_argument('--tinyvit', type=bool, default=False, help='use tiny vit')
    parser.add_argument('--copy', type=bool, default=False, help='use copy')
    args = parser.parse_args()
    return args

def build_model(args):
    # if args.copy:
    #     model = load_model(args.teacher_path, 'cpu')

    # if args.tinyvit:
    #     model = TinyViT(
    #         img_size=1024, in_chans=3, num_classes=1000,
    #         embed_dims=[64, 128, 160, 320],
    #         depths=[2, 2, 6, 2],
    #         num_heads=[2, 4, 5, 10],
    #         window_sizes=[7, 7, 14, 7],
    #         mlp_ratio=4.,
    #         drop_rate=0.,
    #         drop_path_rate=0.0,
    #         use_checkpoint=False,
    #         mbconv_expand_ratio=4.0,
    #         local_conv_size=3,
    #         layer_lr_decay=0.8)    
        
    # else:
    model = AsymmetricCroCo3DStereo(
        pos_embed='RoPE100',
        img_size=(224, 224),
        head_type='dpt',
        output_mode='pts3d', 
        depth_mode=('exp', -inf, inf), 
        conf_mode=('exp', 1, inf), 
        enc_embed_dim=384, 
        enc_depth=12, 
        enc_num_heads=6, 
        dec_embed_dim=768, 
        dec_depth=12, 
        dec_num_heads=12,
        adapter=True)
        
    # pretrained_weights = torch.load("path_to_pth")["model"]
    # model.load_state_dict(pretrained_weights, strict=False)
    return model

def get_optimizer(args, model):
    # train_modules = [model.patch_embed, model.mask_generator, model.rope, model.enc_blocks, model.enc_norm, model.adapter]
    # train_params = torch.nn.ParameterList([p for m in train_modules for p in m.parameters()])
    train_params = list(model.parameters())
    train_params.append(model.mask_token)

    if args.optim == 'adam':
        return optim.Adam(train_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return optim.SGD(train_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return optim.AdamW(train_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    
def customized_mseloss(pred_feats, target_feats):
    return ((pred_feats - target_feats)**2).mean()
    # return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()

def customized_maeloss(pred_feats, target_feats):
    return (torch.abs(pred_feats - target_feats)).mean()
    # return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()

@torch.no_grad()
def test(args, model, teacher, test_loader, l2_loss, l1_loss, device):
    model.eval()
    loss, metric = 0, 0
    for _, batch in enumerate(test_loader):
        for view in batch:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        view1, view2 = batch
        result = model(view1, view2, features_only=True)
        teacher_result = teacher(view1, view2, features_only=True)

        for pred_side, teacher_side in zip(result, teacher_result): # for each view
            pred_feats, _ = pred_side
            target_feats, _ = teacher_side
            loss += l2_loss(pred_feats, target_feats)
            metric += l1_loss(pred_feats, target_feats)

    return loss / len(test_loader) / 2, metric / len(test_loader) / 2

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def main(args):

    # multi gpu settings
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')
    # torch.distributed.init_process_group(backend='nccl')

    # file folder creating
    if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
        os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
    
    if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
        os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # tensorboard
    log_writer = SummaryWriter(log_dir=os.path.join(args.root_path, args.work_dir, args.log_dir))
    
    # dataset
    train_dataset = "Co3d(split='train', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', mask_bg=False, resolution=224, seed=777, features=False)"
    test_dataset = "100 @ Co3d(split='test', ROOT='/ssd1/sa58728/dust3r/data/co3d_subset_processed', mask_bg=False, resolution=224, seed=777, features=False)"
    train_loader = build_dataset(train_dataset, args.batch_size, args.num_workers, test=False)
    test_loader = build_dataset(test_dataset, args.batch_size, args.num_workers, test=False)
    
    # model
    model = build_model(args)
    if CKPT is not None:
        ckpt_new = torch.load(CKPT)
        print(model.load_state_dict(ckpt_new, strict=False))    
    model.to(device)
    # model = load_model(args.teacher_path, device)
    # model.train()
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    teacher = load_model(args.teacher_path, device)
    teacher.eval()
    
    # optimizer and scheduler
    l2_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    total_iters = 0

    for epoch in range(1, args.epochs + 1):
        # new epoch
        print("------start epoch {}------".format(epoch))
        
        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'set_epoch'):
            test_loader.dataset.set_epoch(epoch)
        if hasattr(test_loader, 'sampler') and hasattr(test_loader.sampler, 'set_epoch'):
            test_loader.sampler.set_epoch(epoch)

        # training
        model.train(), teacher.eval()

        for batch_idx, batch in enumerate(train_loader):

            total_iters += 1
            optimizer.zero_grad()

            for view in batch:
                for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                    if name not in view:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)

            view1, view2 = batch
            result = model(view1, view2, features_only=True)
            with torch.no_grad():
                teacher_result = teacher(view1, view2, features_only=True)

            loss, metric = 0, 0
            for pred_side, teacher_side in zip(result, teacher_result): # for each view
                pred_feats, _ = pred_side
                target_feats, _ = teacher_side
                loss += l2_loss(pred_feats, target_feats)
                metric += l1_loss(pred_feats, target_feats)

            loss /= 2
            metric /= 2
            loss.backward()
            optimizer.step()

            # print training info
            if (batch_idx + 1) % args.print_iters == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.1e}\tLoss: {:.6f}\tMetric: {:.4f}'.format(
                    epoch, batch_idx * args.batch_size, len(train_loader.dataset),  # * dist.get_world_size()
                    100. * batch_idx / len(train_loader), scheduler.get_last_lr()[0], loss.item(), metric.item()))
                
                if log_writer is not None:
                    step = int(batch_idx + len(train_loader) * (epoch-1))
                    # We use epoch_1000x as the x-axis in tensorboard. This calibrates different curves when batch size changes.
                    log_writer.add_scalar('train/loss', loss, step)
                    log_writer.add_scalar('lr', scheduler.get_last_lr()[0], step)
            
        # save model
        save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
        print("save model to {}".format(save_path))
        torch.save(model.state_dict(), save_path)

        # evaluation
        test_loss, test_metric = test(args, model, teacher, test_loader, l2_loss, l1_loss, device)
        print('\nTest Loss: {:.4f}, Test Metric: {:.4f}\n'.format(test_loss, test_metric))
        if log_writer is not None:
            step = int(batch_idx + len(train_loader) * (epoch-1))
            # We use epoch_1000x as the x-axis in tensorboard. This calibrates different curves when batch size changes.
            log_writer.add_scalar('test/loss', test_loss, step)
            log_writer.add_scalar('test/metric', test_metric, step)

        # dist.barrier()
        scheduler.step()

        if log_writer is not None:
            log_writer.flush()

    # save final model
    torch.save(model.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=True,
                             drop_last=False)

    print(f"{split} dataset length: ", len(loader))
    return loader


if __name__ == "__main__":
    args = parse_option()
    main(args)

