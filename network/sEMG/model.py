import torch.nn as nn
import torch
from network.sEMG.df import ConditionalModel
from network.sEMG.df_t import ConditionalModel_t

class Model(nn.Module):
    def __init__(self, timesteps=1000, feats=128):
        super().__init__()
        self.timesteps = timesteps
        self.model = ConditionalModel(feats=feats)
        self.model_t = ConditionalModel_t(feats=feats)

    def forward(self, x, noise_adder, clean_x):
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        max_t = t.max()

        batch_noisy_x = torch.zeros_like(x)
        for i in range(max_t + 1):
            mask = (t == i).unsqueeze(1).unsqueeze(2)
            if mask.any():
                batch_noisy_x = torch.where(mask, x, batch_noisy_x)
            noise = noise_adder.get_train_noise(clean_x)
            x = x + noise

        x = batch_noisy_x
        pred_clean_x = self.model(x, t)
        pred_t = self.model_t(x)
        gt_t = t.float() / self.timesteps

        return pred_clean_x, pred_t, gt_t

    @torch.no_grad()
    def sampling(self, x):
        pred_t = self.model_t(x)
        t = (pred_t * (self.timesteps - 1)).long()
        t_max = t.max()

        x_denoise_dict = {}
        x_denoise = torch.zeros_like(x)
        for i in range(t_max, -1, -1):
            mask = (t == i).unsqueeze(1).unsqueeze(2)
            if mask.any():
                x_denoise = torch.where(mask, x, x_denoise)
            batch_t = torch.full((x.shape[0],), i, device=x.device)
            x_denoise = self.model(x_denoise, batch_t)
            x_denoise_dict[i] = x_denoise

        return x_denoise

