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
from dataset.dataset_sEMG import EMGDataset, EMGTestDataset
from utils.utils_sEMG import AugmentEMGNoise
from eval.eval_sEMG import cal_snr, cal_rmse, cal_prd, cal_ARV, cal_KR, cal_MF, cal_R2, cal_CC
from network.sEMG.model import Model


# Training function
def train(model, device, train_loader, optimizer, scheduler, epoch, debug, noise_adder, loss_fn, rank):
    model.train()
    train_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for i, (clean_data, noisy_data) in progress_bar:
        clean_data = clean_data.to(device)
        noisy_data = noisy_data.to(device)

        pred_clean_data, pred_t, gt_t = model(noisy_data, noise_adder, clean_data)

        optimizer.zero_grad()
        loss_noise = loss_fn(pred_clean_data, clean_data)
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

# Evaluation function
def evaluate(model, device, test_loader, debug, noise_adder, loss_fn, rank):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    test_loss = 0
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Validation")
    with torch.no_grad():
        snr_list = {}
        loss_list = {}
        rmse_list = {}
        prd_list = {}
        arv_list = {}
        kr_list = {}
        mf_list = {}
        r2_list = {}
        cc_list = {}
        for i, (clean_data, noisy_data, snr_batch, sti_batch, file_name_batch) in progress_bar:
            clean_data = clean_data.to(device)
            noisy_data = noisy_data.to(device)

            denoise_data = model.sampling(noisy_data)
            if torch.isnan(denoise_data).any() or torch.isinf(denoise_data).any():
                continue

            img_loss = loss_fn(denoise_data, clean_data)
            test_loss += img_loss.item()

            clean_data = clean_data.cpu().detach().numpy()
            denoise_data = denoise_data.cpu().detach().numpy()
            noisy_data = noisy_data.cpu().detach().numpy()
            snr_batch = np.array(snr_batch)
            sti_batch = sti_batch.cpu().detach().numpy()
            for i, (pred_i, clean, noisy, snr, sti, file_name) in enumerate(zip(denoise_data, clean_data, noisy_data, snr_batch, sti_batch, file_name_batch)):
                clean = clean.squeeze().squeeze()
                enhanced = pred_i.squeeze().squeeze()
                noisy = noisy.squeeze().squeeze()
                noise_gt = noisy - clean
                noise_pred = noisy - enhanced
                sti = sti.squeeze().squeeze()
                loss = loss_fn(torch.from_numpy(enhanced), torch.from_numpy(clean)).item()
                SNR = cal_snr(clean, enhanced)
                RMSE = cal_rmse(clean, enhanced)
                PRD = cal_prd(clean, enhanced)
                RMSE_ARV = cal_rmse(cal_ARV(clean), cal_ARV(enhanced))
                KR = abs(cal_KR(clean) - cal_KR(enhanced))
                MF = cal_rmse(cal_MF(clean, sti), cal_MF(enhanced, sti))
                R2 = cal_R2(clean, enhanced)
                CC = cal_CC(clean, enhanced)

                snr_list[snr] = snr_list.get(snr, []) + [SNR]
                loss_list[snr] = loss_list.get(snr, []) + [loss]
                rmse_list[snr] = rmse_list.get(snr, []) + [RMSE]
                prd_list[snr] = prd_list.get(snr, []) + [PRD]
                arv_list[snr] = arv_list.get(snr, []) + [RMSE_ARV]
                kr_list[snr] = kr_list.get(snr, []) + [KR]
                mf_list[snr] = mf_list.get(snr, []) + [MF]
                r2_list[snr] = r2_list.get(snr, []) + [R2]
                cc_list[snr] = cc_list.get(snr, []) + [CC]


            progress_bar.set_postfix(loss=test_loss / (i + 1))
            if not debug and rank == 0:
                wandb.log({"test_loss": test_loss / (i + 1), "batch_idx": i})

    test_loss /= len(test_loader)
    if rank == 0:
        print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    if not debug and rank == 0:
        wandb.log({"test_loss": test_loss})
        for snr in snr_list.keys():
            avg_loss = sum(loss_list[snr]) / len(loss_list[snr])
            avg_snr = sum(snr_list[snr]) / len(snr_list[snr])
            avg_rmse = sum(rmse_list[snr]) / len(rmse_list[snr])
            avg_prd = sum(prd_list[snr]) / len(prd_list[snr])
            avg_arv = sum(arv_list[snr]) / len(arv_list[snr])
            avg_kr = sum(kr_list[snr]) / len(kr_list[snr])
            avg_mf = sum(mf_list[snr]) / len(mf_list[snr])
            avg_r2 = sum(r2_list[snr]) / len(r2_list[snr])
            avg_cc = sum(cc_list[snr]) / len(cc_list[snr])
            print("======= SNR: {} ========".format(snr))
            print("Loss: {:.4f}, rmse: {:.4f}, PRD: {:.4f},".format(avg_loss, avg_rmse, avg_prd))
            print("ARV: {:.4f}, KR: {:.4f}, MF: {:.4f}".format(avg_arv, avg_kr, avg_mf))
            print("R2: {:.4f}, CC: {:.4f}".format(avg_r2, avg_cc))
            wandb.log({f"Loss_{snr}": avg_loss, f"SNR_{snr}": avg_snr, f"RMSE_{snr}": avg_rmse})
            wandb.log({f"PRD_{snr}": avg_prd, f"ARV_{snr}": avg_arv, f"KR_{snr}": avg_kr})
            wandb.log({f"MF_{snr}": avg_mf, f"R2_{snr}": avg_r2, f"CC_{snr}": avg_cc})
    return test_loss

def main_worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)


    train_snr_list = [-5, -7, -9, -11, -13, -15]
    valid_snr_list = [-5, -7, -9, -11, -13, -15]
    test_snr_list = [0, -2, -4, -6, -8, -10, -12, -14]

    result_dir = os.path.join('results', f"{args.alias}")
    os.makedirs(result_dir, exist_ok=True)

    if not args.debug and rank == 0:
        wandb.init(project="emg", config=args, name=f"{args.alias}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainset = EMGDataset(train_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(dataset=trainset, sampler=train_sampler, num_workers=0, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=True)

    valset = EMGTestDataset(val_dir)
    val_dataloader = DataLoader(dataset=valset, num_workers=0, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=False)
    
    test_dataset = EMGTestDataset(test_dir)
    test_dataloader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=False)

    noise_adder_train = AugmentEMGNoise(noise_train_dir, train_snr_list, device)
    noise_adder_val = AugmentEMGNoise(noise_val_dir, valid_snr_list, device)
    noise_adder_test = AugmentEMGNoise(noise_test_dir, test_snr_list, device)

    model = Model(timesteps=args.timesteps)
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_dataloader), pct_start=0.25, anneal_strategy='cos')
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
        print(f"Loading model checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        loss = train(model, device, train_dataloader, optimizer, scheduler, epoch, args.debug, noise_adder_train, loss_fn, rank)
        if rank == 0:
            evaluate(model, device, test_dataloader, args.debug, noise_adder_test, loss_fn, rank)
    dist.destroy_process_group()

# Main function
def main():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, default='', help='resume training from checkpoint')
    parser.add_argument('--ckpt', type=str, default='', help='load model checkpoint')
    parser.add_argument('--debug', action='store_true', default=True, help='enable debug mode')
    parser.add_argument('--gpus', type=str, default='3', help='Comma separated list of GPU ids to use (default: 0)')
    parser.add_argument('--alias', type=str, default='experiment', help='alias for the project')
    parser.add_argument('--timesteps', type=int, default=100, help='iterations for model')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'

    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

if __name__ == '__main__':
    main()