from torch.utils.data import DataLoader
from dataset.dataset_msa import Dataset_msa, msa_dataset_collate
import torch
from network.msa.model import Model
import numpy as np
from tqdm import tqdm
import os, time
import wandb
import warnings
warnings.filterwarnings("ignore")
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import sys
import esm
from eval.eval_msa import eval_msa



def train(device='cpu', resume=False, checkpoint=None, **kwargs):

    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval().to(device)
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

    dataset = Dataset_msa(train_Datapath, data_num="all")
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=msa_dataset_collate, drop_last=True)
    testset = Dataset_msa(val_Datapath, data_num=50)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=msa_dataset_collate)

    timesteps = 1000
    model=Model(
        device=device,
        timesteps=timesteps
    ).to(device)
    print('model construct completed')

    start_epoch = 0
    total_epoch = 10000
    train_total_step = 0
    val_total_step = 0
    best_auc = 0
    auc = 0
    min_loss = 1000

    optimizer=AdamW(model.parameters(),lr=0.0001)
    lr_scheduler=OneCycleLR(optimizer,0.0001,total_steps=total_epoch*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.CrossEntropyLoss(ignore_index=-1)
    t_loss_fn = nn.MSELoss()
    print('optimizer construct completed')

    current_time = time.strftime("%m_%d_%H_%M", time.localtime())
    Log_dir = f'log/esm-{current_time}/'

    wandb.init(project='esm', name=f'esm-{current_time}')

    if checkpoint is not None and checkpoint != '':
        model.load_state_dict(torch.load(checkpoint), strict=False)

    for epoch in range(start_epoch, total_epoch):
        if epoch == start_epoch:
            auc = evaluate_msa(
                model, test_dataloader, device, device, epoch,
                msa_transformer, msa_transformer_batch_converter, debug=False
            )
            torch.cuda.empty_cache()

        train_loss = train_msa(
            model, train_dataloader, optimizer, lr_scheduler, loss_fn,
            t_loss_fn, device, device, epoch, debug=False
        )

        os.makedirs(Log_dir,exist_ok=True)
        torch.save(model.state_dict(), Log_dir + f'lastest.pth')
        torch.cuda.empty_cache()

        if epoch % 1 == 0:
            auc = evaluate_msa(
                model, test_dataloader, device, device, epoch,
                msa_transformer, msa_transformer_batch_converter, debug=False
            )

        if auc > best_auc:
            best_auc = auc
            os.makedirs(Log_dir,exist_ok=True)
            torch.save(model.state_dict(), Log_dir + f'best_model.pth')
        torch.cuda.empty_cache()

def train_msa(model, dataloader, optimizer, lr_scheduler, loss_fn, t_loss_fn, device, epoch, debug=False):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    for i, (good_tokens, bad_tokens, full_tokens, msa, name, msa_array) in progress_bar:
        good_tokens = [match.to(device) for match in good_tokens]
        bad_tokens = [match.to(device) for match in bad_tokens]
        full_tokens = [match.to(device) for match in full_tokens]

        pred_label, gt_label, pred_t, gt_t = model(
            good_tokens, bad_tokens, full_tokens, msa, name, msa_array, training=True
        )
        label_loss = loss_fn(pred_label, gt_label)
        t_loss = t_loss_fn(pred_t, gt_t)
        loss = label_loss + t_loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % 10 == 0:
            progress_bar.set_postfix(loss=loss.item(), loss_avg=train_loss / (i + 1))
            if not debug:
                wandb.log({"loss": loss.item(), "epoch": epoch, "batch_idx": i})

    train_loss /= len(dataloader)
    if not debug:
        wandb.log({"train_loss": train_loss})

    return train_loss

def evaluate_msa(model, dataloader, device, epoch, msa_transformer, msa_transformer_batch_converter, debug=False):
    model.eval()
    AUC_list = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
    with torch.no_grad():
        for step, (good_tokens, bad_tokens, full_tokens, msa, name, msa_array) in progress_bar:
            good_tokens = [match.to(device) for match in good_tokens]
            bad_tokens = [match.to(device) for match in bad_tokens]
            full_tokens = [match.to(device) for match in full_tokens]

            denoised_msa_dic = model(
                good_tokens, bad_tokens, full_tokens, msa, name, msa_array, training=False
            )
            auc = eval_msa(denoised_msa_dic, device, msa_transformer, msa_transformer_batch_converter)
            AUC_list.append(auc)
            progress_bar.set_postfix(PL=auc)

            if not debug:
                wandb.log({"Step_PL": auc})

    MAUC = np.mean(AUC_list) if len(AUC_list) > 0 else 0.0
    if not debug:
        wandb.log({"PL": MAUC})
    print("PL:", MAUC)

    return MAUC


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    train(device=device, device=device, resume=False, checkpoint='')