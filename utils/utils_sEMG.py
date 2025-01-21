import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

class AugmentEMGNoise(object):
    def __init__(self, noise_train_dir, train_snr_list, device):
        self.device = device
        self.noise_file_names = glob(os.path.join(noise_train_dir, "*.npy"), recursive=True)
        self.noise_data = np.array([np.load(file_name) for file_name in self.noise_file_names])
        self.noise_data = torch.tensor(self.noise_data, dtype=torch.float32).to(device).unsqueeze(1)
        self.train_snr_list = train_snr_list
        self.clean_rate = 1000

    def add_train_noise(self, y_clean):
        normalize = False
        bs = y_clean.shape[0]
        SNR = random.choices(self.train_snr_list, k=bs)
        SNR = torch.tensor(np.array(SNR)).to(self.device)
        noise_ori = self.noise_data[torch.randint(0, len(self.noise_data), (bs,))]
        if noise_ori.shape[2] < y_clean.shape[2]:
            tmp = (len(y_clean) // len(noise_ori)) + 1
            y_noise = []
            for _ in range(tmp):
                y_noise.extend(noise_ori)
        else:
            y_noise = noise_ori
        start = torch.randint(0, y_noise.shape[2]-y_clean.shape[2], (bs,)).to(self.device)
        tmp = []
        for i in range(bs):
            tmp.append(y_noise[i, :, start[i]:start[i]+y_clean.shape[2]])
        y_noise = torch.stack(tmp)
        del tmp
        y_clean_pw = torch.einsum('ijk,ijk->i', y_clean, y_clean)
        y_noise_pw = torch.einsum('ijk,ijk->i', y_noise, y_noise)
        scalar = torch.sqrt(y_clean_pw / (torch.pow(10.0, SNR / 10.0) * y_noise_pw))
        noise = scalar.unsqueeze(1).unsqueeze(2) * y_noise
        y_noisy = y_clean + noise
        if normalize: 
            norm_scalar = np.max(abs(y_noisy))
            y_noisy = y_noisy/norm_scalar
        return y_noisy

    def get_train_noise(self, y_clean):
        bs, _, clean_len = y_clean.shape
        SNR = torch.tensor(random.choices(self.train_snr_list, k=bs)).to(self.device)
        idx = torch.randint(0, len(self.noise_data), (bs,))
        noise_ori = self.noise_data[idx]
        _, _, noise_len = noise_ori.shape
        if noise_len < clean_len:
            repeat_times = (clean_len - 1) // noise_len + 1
            noise_ori = noise_ori.repeat(1, 1, repeat_times)
            noise_len = noise_ori.shape[2]
        max_start = noise_len - clean_len
        start = torch.randint(0, max_start + 1, (bs,))
        y_noise = torch.stack([noise_ori[i, :, start[i]:start[i]+clean_len] for i in range(bs)])
        y_clean_pw = torch.sum(y_clean ** 2, dim=(1, 2))
        y_noise_pw = torch.sum(y_noise ** 2, dim=(1, 2))
        scalar = torch.sqrt(y_clean_pw / (torch.pow(10.0, SNR / 10.0) * y_noise_pw))
        noise = scalar.view(bs, 1, 1) * y_noise
        return noise

def plot_enhanced_signal(enhanced, file_name):
    x = np.arange(10000)
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2)
    colors = sns.color_palette("husl", n_colors=2)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    plt.plot(x, enhanced, color=colors[0], linewidth=1.5, alpha=0.8)
    plt.xlim(0, 10000)
    plt.ylim(-1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def plot_clean_enhanced_comparison(clean, enhanced, file_name):
    x = np.arange(10000)
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2)
    colors = sns.color_palette("husl", n_colors=2)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    plt.plot(x, clean, color=colors[0], label='Clean', linewidth=1.5, alpha=0.8)
    plt.plot(x, enhanced, color=colors[1], label='Enhanced', linewidth=1.5, alpha=0.8)
    plt.xlim(0, 10000)
    plt.ylim(-1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def plot_clean_enhanced_noise_comparison(clean, enhanced, noise, file_name):
    x = np.arange(10000)
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2)
    colors = [
        (240/255, 189/255, 94/255),
        (165/255, 165/255, 165/255),
        (222/255, 125/255, 116/255)
    ]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    plt.plot(x, clean, color=colors[0], label='Clean', linewidth=1.5, alpha=0.8)
    plt.plot(x, noise, color=colors[1], label='Noise', linewidth=1.5, alpha=0.8)
    plt.plot(x, enhanced, color=colors[2], label='Denoised', linewidth=1.5, alpha=0.8)
    plt.xlim(0, 10000)
    plt.ylim(-1, 1)
    ax.tick_params(axis='both', which='major', labelsize=46)
    ax.set_xticks([0, 10000])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', facecolor='white')
    plt.close()