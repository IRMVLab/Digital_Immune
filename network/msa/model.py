import torch
from torch import nn
from network.msa.transformer import PointTransformer
from network.msa.dgcnn import DGCNN_cls
import math
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, device, timesteps):
        super().__init__()

        self.timesteps = timesteps
        self.in_channels = 2
        self.w = 1
        self.h = 1
        self.max_outliers = 128
        self.max_msas = 256
        self.sample_steps = 30


        self.DGCNN_cls = DGCNN_cls()
        self.network = PointTransformer(device)

    def forward(self, good_tokens, bad_tokens, full_tokens, msa, name, msa_array, training=True):
        device = good_tokens[0].device
        bs = len(good_tokens)

        if training:
            t = torch.randint(0, self.timesteps, (bs,)).to(device)
            outlier_ratio = self.sqrt_one_minus_alphas_cumprod.gather(-1, t)

            labels = []
            datanums = []
            inputs = []
            pad_mask = torch.ones(bs, self.max_msas).to(device)
            for i in range(bs):
                good_tokens_num = good_tokens[i].shape[0]
                outlier_num = int(self.max_outliers * outlier_ratio[i])
                shuffle = torch.randperm(bad_tokens[i].shape[0])
                bad_tokens[i] = bad_tokens[i][shuffle, :]
                noise_matches = bad_tokens[i][:outlier_num]
                good_tokens[i] = torch.cat((good_tokens[i], noise_matches), dim=0)
                if good_tokens[i].shape[0] > self.max_msas:
                    good_tokens[i] = good_tokens[i][:self.max_msas, :]
                assert good_tokens[i].shape[0] <= self.max_msas

                mask = torch.zeros(good_tokens[i].shape[0]).to(device)
                mask[good_tokens_num:] = 1.
                shuffle = torch.randperm(good_tokens[i].shape[0])
                good_tokens[i] = good_tokens[i][shuffle, :]
                mask = mask[shuffle]
                inputs.append(good_tokens[i])

                datanum = good_tokens[i].shape[0]
                labels.append(mask)
                datanums.append(datanum)
                pad_mask[i, :datanum] = 1.

            inputs = pad_sequence(inputs, batch_first=True, padding_value=1)
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)

            padding_needed = self.max_msas - inputs.shape[1]
            inputs = F.pad(inputs, (0, 0, 0, padding_needed), "constant", 1).permute(0, 2, 1)
            gt_label = F.pad(labels, (0, padding_needed), "constant", -1).long()

            datanums = torch.tensor(datanums).to(device)
            pred_t = self.DGCNN_cls(inputs)
            gt_t = t / float(self.timesteps)
            pred_label = self.network(inputs.int(), t)
            
            return pred_label, gt_label, pred_t, gt_t

        else:
            inputs = []
            for i in range(bs):
                inputs.append(good_tokens[i])

            msas_process = []

            inputs_batch = pad_sequence(inputs, batch_first=True, padding_value=1)
            padding_needed = self.max_msas - inputs_batch.shape[1]
            inputs_batch = F.pad(inputs_batch, (0, 0, 0, padding_needed), "constant", 1).permute(0, 2, 1)

            pred_t = self.DGCNN_cls(inputs_batch)
            pred_t = torch.round(pred_t * self.timesteps).long()

            tokens_num = []
            for i in range(inputs_batch.shape[0]):
                tokens_num.append(inputs[i].shape[0])
                msas_process.append([inputs[i]])

            for i in range(inputs_batch.shape[0]):
                inputs_clean = inputs_batch[i][:, :tokens_num[i]]
                inputs_batch_i = inputs_batch[i].unsqueeze(0)
                for t in reversed(range(0, pred_t[i])):
                    t_tensor = torch.tensor([t]).to(device)
                    pred_label = self.network(inputs_batch_i.int(), t_tensor)
                    pred_label = torch.argmax(pred_label, dim=1).bool()
                    pred_label = pred_label.squeeze(0)
                    xxx = ~pred_label[:tokens_num[i]]
                    sum = xxx.sum()
                    inputs_clean = inputs_clean[:, ~pred_label[:tokens_num[i]]].T

                    if 0:
                        outlier_ratio = self.alphas_cumprod.gather(-1, t_tensor)
                        outlier_ratio = 1. - outlier_ratio

                        outlier_num = int(self.max_outliers * outlier_ratio)
                        shuffle = torch.randperm(bad_tokens[i].shape[0])
                        bad_tokens[i] = bad_tokens[i][shuffle, :]
                        noise_matches = bad_tokens[i][:outlier_num]
                        inputs_clean = torch.cat((inputs_clean, noise_matches), dim=0)

                        if inputs_clean.shape[0] > self.max_msas:
                            inputs_clean = inputs_clean[:self.max_msas, :]
                        assert inputs_clean.shape[0] <= self.max_msas
                        shuffle = torch.randperm(inputs_clean.shape[0])
                        inputs_clean = inputs_clean[shuffle, :].T

                        tokens_num[i] = inputs_clean.shape[1]
                        msas_process[i].append(inputs_clean.T)
                    else:
                        inputs_clean = inputs_clean.T
                        tokens_num[i] = inputs_clean.shape[1]
                        msas_process[i].append(inputs_clean.T)

                    if msas_process[i][-1].shape[0] < 1:
                        break
                    inputs_batch_i = pad_sequence([msas_process[i][-1]], batch_first=True, padding_value=1)
                    padding_needed = self.max_msas - inputs_batch_i.shape[1]
                    inputs_batch_i = F.pad(inputs_batch_i, (0, 0, 0, padding_needed), "constant", 1).permute(0, 2, 1)

            denoised_msa_dic = {}
            for i in range(len(msas_process)):
                msa_clean = msas_process[i][len(msas_process[i]) - 1]
                if msa_clean.shape[0] < 1:
                    msa_clean = msas_process[i][len(msas_process[i]) - 2]
                indices = []
                for row in msa_clean:
                    matched_indices = (full_tokens[i] == row).all(dim=1).nonzero(as_tuple=True)[0]
                    indices.append(matched_indices)
                indices = torch.cat(indices)
                indices = torch.unique(indices)
                indices = torch.sort(indices)[0]
                if indices[0] != 0:
                    indices = torch.cat([torch.tensor([0]).to(device), indices])
                indices = indices.cpu().numpy()
                denoised_msa = [seq for idx, seq in enumerate(msa[i]) if idx in indices]
                denoised_msa_dic[name[i]] = denoised_msa

            return denoised_msa_dic

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas