import torch.nn as nn
import torch
from network.pic.unet import Unet
from network.pic.unet_t import Unet_t

class Model(nn.Module):
    def __init__(self, image_size, in_channels, time_embedding_dim=256, timesteps=1000, base_dim=32, dim_mults=[1, 2, 4, 8]):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        self.model = Unet(timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults)
        self.model_t = Unet_t(in_channels, base_dim, dim_mults)

    def forward(self, x, noise_adder, ori_image):
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        max_t = t.max()
        batch_noisy_x = torch.zeros_like(x)
        batch_noise = torch.zeros_like(x)
        noise = torch.zeros_like(x)
        for i in range(max_t + 1):
            mask = (t == i).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            if mask.any():
                batch_noisy_x = torch.where(mask, x, batch_noisy_x)
                batch_noise = torch.where(mask, noise, batch_noise)
            _, noise = noise_adder.add_train_noise(ori_image)
            x = x + noise

        x = batch_noisy_x
        noise = batch_noise
        x = x * 2. - 1.
        pred_clean_x = self.model(x, t)
        pred_clean_x = (pred_clean_x + 1.) / 2.
        pred_t = self.model_t(x)
        gt_t = t.float() / self.timesteps

        return pred_clean_x, pred_t, gt_t

    @torch.no_grad()
    def sampling(self, x):
        pred_t = self.model_t(x * 2. - 1.)
        t = (pred_t * (self.timesteps - 1)).long()
        t_max = t.max()
        x_denoise_dict = {}
        x_denoise = torch.zeros_like(x)
        for i in range(t_max, -1, -1):
            mask = (t == i).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            if mask.any():
                x_denoise = torch.where(mask, x, x_denoise)
            batch_t = torch.full((x.shape[0],), i, device=x.device)
            x_denoise = x_denoise * 2. - 1.
            x_denoise = self.model(x_denoise, batch_t)
            x_denoise = (x_denoise + 1.) / 2.
            x_denoise_dict[i] = x_denoise

        return x_denoise

