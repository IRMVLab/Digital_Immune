import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import wandb
import os
import numpy as np
import random
from tqdm import tqdm
from dipy.io.image import save_nifti

from dataset.dataset_mri import MRIDataset
from utils.utils_mri import AugmentNoise
from network.mri.model import Model


def calculate_psnr(target, ref, data_range=255.0):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr

def train(model, device, train_loader, optimizer, scheduler, epoch, noise_adder, loss_fn):
    model.train()
    train_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for i, train_data in progress_bar:
        train_data['X'] = train_data['X'].to(device)
        image = train_data['X']
        noise = noise_adder.get_train_noise(image)
        pred = model(image, noise)

        optimizer.zero_grad()
        loss = loss_fn(pred, noise)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0:
            progress_bar.set_postfix(loss=loss.item(), loss_avg=train_loss / (i + 1), lr=scheduler.get_last_lr()[0])
            wandb.log({"loss": loss.item(), "epoch": epoch, "batch_idx": i, "lr": scheduler.get_last_lr()[0]})
            wandb.log({"train_loss": train_loss / (i + 1)})
    train_loss /= len(train_loader)
    print(f'\nTrain set: Average loss: {train_loss:.4f}')
    wandb.log({"train_loss": train_loss})
    return loss

def evaluate(model, device, val_loader, noise_adder, loss_fn):
    model.eval()
    val_loss = 0
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
    with torch.no_grad():
        for i, val_data in progress_bar:
            val_data['X'] = val_data['X'].to(device)
            image = val_data['X']
            noise = noise_adder.get_train_noise(image)
            pred = model(image, noise)
            img_loss = loss_fn(pred, noise)
            val_loss += img_loss.item()
            progress_bar.set_postfix(loss=val_loss / (i + 1))
            wandb.log({"val_loss": val_loss / (i + 1), "batch_idx": i})
    val_loss /= len(val_loader)
    print(f'\nValidation set: Average loss: {val_loss:.4f}\n')
    wandb.log({"val_loss": val_loss})
    return val_loss

def main_worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)

    wandb.init(project="mri", config=args, name=f"{args.alias}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_dataset = MRIDataset(dataroot=args.dataroot, phase='train', padding=3, in_channel=1)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, num_workers=0, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    val_dataset = MRIDataset(dataroot=args.dataroot, phase='val', padding=3, in_channel=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False)

    noise_adder = AugmentNoise(style="gauss45")

    model = Model(timesteps=args.timesteps, in_channels=1, base_dim=args.model_base_dim, dim_mults=[2, 4])
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')
    loss = 0.0

    start_epoch = 1
    if args.resume != '' and args.resume is not None:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

    if args.ckpt != '' and args.ckpt is not None:
        print(f"loading model from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        loss = train(model, device, train_loader, optimizer, scheduler, epoch, noise_adder, loss_fn)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(result_dir, 'model_checkpoint.pth'))
            evaluate(model, device, val_loader, noise_adder, loss_fn)
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--batch-size', type=int, default=12, help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, default='', help='resume training from checkpoint')
    parser.add_argument('--ckpt', type=str, default='', help='resume training from checkpoint')
    parser.add_argument('--gpus', type=str, default='0', help='Comma separated list of GPU ids to use (default: 0)')
    parser.add_argument('--alias', type=str, default='experiment', help='alias for the project')
    parser.add_argument('--image_size', type=int, default=256, help='size of the input image')
    parser.add_argument('--timesteps', type=int, default=1000, help='iterations for model')
    parser.add_argument('--model_base_dim', type=int, default=64, help='base dim of Unet')
    parser.add_argument('--dataroot', type=str, default='', help='path to dataset')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

if __name__ == '__main__':
    main()