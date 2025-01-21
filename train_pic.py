import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import argparse
import wandb
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2

from dataset.dataset_pic import Imagenet_val, ValidationDataset, ValidationDatasetBSD300
from utils.utils_pic import AugmentNoise
from network.pic.model import Model

def train(model, device, train_loader, optimizer, scheduler, epoch, debug, noise_adder_collect, noise_adder, loss_fn, rank):
    model.train()
    train_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for i, (ori_image, ori_sizes) in progress_bar:
        ori_image = ori_image.to(device)
        noisy_image = ori_image.clone()
        noisy_image, _ = noise_adder_collect.add_train_noise(noisy_image)
        pred_clean_image, pred_t, gt_t = model(noisy_image, noise_adder, ori_image)

        optimizer.zero_grad()
        loss_noise = loss_fn(pred_clean_image, ori_image)
        loss_t = loss_fn(pred_t, gt_t)
        loss = loss_noise + loss_t
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0:
            progress_bar.set_postfix(loss_noise=loss_noise.item(), loss_t=loss_t.item(), loss=loss.item(), loss_avg=train_loss / (i + 1), lr=scheduler.get_last_lr()[0])
            if not debug and rank == 0:
                wandb.log({"loss": loss.item(), "epoch": epoch, "batch_idx": i, "lr": scheduler.get_last_lr()[0]})
                wandb.log({"loss_noise": loss_noise.item(), "loss_t": loss_t.item(), "train_loss": train_loss / (i + 1)})
    train_loss /= len(train_loader)
    if rank == 0:
        print(f'\nTrain set: Average loss: {train_loss:.4f}')
    if not debug and rank == 0:
        wandb.log({"train_loss": train_loss})
    
    return loss

def evaluate(model, device, test_loader, debug, noise_adder_collect, loss_fn, rank):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    test_loss = 0
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Validation")
    with torch.no_grad():
        ori_images = []
        denoise_images = []
        noisy_images = []
        ws = []
        hs = []
        for i, (ori_image, ori_size) in progress_bar:
            ori_image = ori_image.to(device)
            noisy_image = ori_image.clone()
            noisy_image, _ = noise_adder_collect.add_train_noise(noisy_image)
            denoise_image = model.sampling(noisy_image)
            ws.append(ori_size[1])
            hs.append(ori_size[0])

            ori_images.append(ori_image)
            denoise_images.append(denoise_image)
            noisy_images.append(noisy_image)

            img_loss = loss_fn(denoise_image, ori_image)
            test_loss += img_loss.item()
            progress_bar.set_postfix(loss=test_loss / (i + 1))
            if not debug and rank == 0:
                wandb.log({"test_loss": test_loss / (i + 1), "batch_idx": i})
        ori_images = torch.cat(ori_images, dim=0)
        denoise_images = torch.cat(denoise_images, dim=0)
        noisy_images = torch.cat(noisy_images, dim=0)

        ws = torch.cat(ws, dim=0)
        hs = torch.cat(hs, dim=0)
    
    for i in range(ori_images.size(0)):
        ori_image = ori_images[i].cpu().detach().numpy().transpose(1, 2, 0)
        denoise_image = denoise_images[i].cpu().detach().numpy().transpose(1, 2, 0)
        noisy_image = noisy_images[i].cpu().detach().numpy().transpose(1, 2, 0)
        ori_image = np.clip(ori_image * 255.0 + 0.5, 0, 255).astype(np.uint8)
        denoise_image = np.clip(denoise_image * 255.0 + 0.5, 0, 255).astype(np.uint8)
        noisy_image = np.clip(noisy_image * 255.0 + 0.5, 0, 255).astype(np.uint8)
        w = ws[i].item()
        h = hs[i].item()

        ori_image = cv2.resize(ori_image, (w, h))
        denoise_image = cv2.resize(denoise_image, (w, h))
        noisy_image = cv2.resize(noisy_image, (w, h))
        cv2.imwrite(f"results/{i}_ori.png", ori_image)
        cv2.imwrite(f"results/{i}_denoise.png", denoise_image)
        cv2.imwrite(f"results/{i}_noisy_1.png", noisy_image)

    test_loss /= len(test_loader)
    if rank == 0:
        print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    if not debug and rank == 0:
        wandb.log({"test_loss": test_loss})
    return test_loss

def main_worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)


    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)

    if not args.debug and rank == 0:
        wandb.init(project="pic", config=args, name=f"{args.alias}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    TrainingDataset = Imagenet_val(train_dir, image_size=args.image_size, data_num='all')
    train_sampler = torch.utils.data.distributed.DistributedSampler(TrainingDataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(dataset=TrainingDataset, sampler=train_sampler, num_workers=0, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    Kodak_dir = os.path.join(val_dir, "Kodak")
    BSD300_dir = os.path.join(val_dir, "BSD300")

    valid_dict = {
        "Kodak24": DataLoader(ValidationDataset(Kodak_dir, image_size=args.image_size), batch_size=args.test_batch_size, pin_memory=True, shuffle=False),
        "BSD300": DataLoader(ValidationDatasetBSD300(BSD300_dir, image_size=args.image_size), batch_size=args.test_batch_size, pin_memory=True, shuffle=False),
    }

    noise_adder_collect = AugmentNoise('poisson30')
    noise_adder = AugmentNoise('poisson30')

    model = Model(timesteps=args.timesteps, image_size=args.image_size, in_channels=3, base_dim=args.model_base_dim, dim_mults=[2,4])
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs*len(train_dataloader), pct_start=0.25, anneal_strategy='cos')
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
        loss = train(model, device, train_dataloader, optimizer, scheduler, epoch, args.debug, noise_adder_collect, noise_adder, loss_fn, rank)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(result_dir, 'model_checkpoint.pth'))
            evaluate(model, device, valid_dict["Kodak24"], args.debug, noise_adder_collect, loss_fn, rank)
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--batch-size', type=int, default=12, help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, default='', help='resume training from checkpoint')
    parser.add_argument('--ckpt', type=str, default='', help='resume training from checkpoint')
    parser.add_argument('--debug', action='store_true', default=False, help='enable debug mode')
    parser.add_argument('--gpus', type=str, default='3', help='Comma separated list of GPU ids to use (default: 0)')
    parser.add_argument('--alias', type=str, default='experiment', help='alias for the project')
    parser.add_argument('--image_size', type=int, default=256, help='size of the input image')
    parser.add_argument('--timesteps', type=int, default=100, help='iterations for model')
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

if __name__ == '__main__':
    main()