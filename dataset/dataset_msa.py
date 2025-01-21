import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset_msa(Dataset):
    def __init__(self, Datapath, data_num='all', converter=None):
        self.Datapath_pkl = Datapath + 'seq_pkl'
        self.Datapath_tokens = Datapath + 'seq_tokens'
        self.data_num = data_num
        self.pairs = os.listdir(self.Datapath_tokens)
        self.pairs.sort()
        self.IDS = []
        for i in range(len(self.pairs)):
            id = os.path.splitext(self.pairs[i])[0]
            if id not in self.IDS:
                self.IDS.append(id)
        self.max_tokens = 256

    def __getitem__(self, idx):
        id = self.IDS[idx]
        dic = pickle.load(open(os.path.join(self.Datapath_pkl, id+'.pkl'), 'rb'))
        good_id = dic['good_idx']
        bad_id = dic['bad_idx']
        random.shuffle(good_id)
        random.shuffle(bad_id)
        msa = dic['msa']
        msa_tokens = torch.load(os.path.join(self.Datapath_tokens, id+'.pt'))[0]

        if msa_tokens.shape[1] < self.max_tokens:
            padding = torch.ones(msa_tokens.shape[0], self.max_tokens - msa_tokens.shape[1])
            msa_tokens = torch.cat([msa_tokens, padding], dim=-1)

        good_tokens = msa_tokens[good_id]
        bad_tokens = msa_tokens[bad_id]
        full_tokens = msa_tokens
        name = id
        msa_array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

        return good_tokens, bad_tokens, full_tokens, msa, name, msa_array

    def __len__(self):
        if self.data_num == 'all':
            return len(self.pairs)
        else:
            return self.data_num

def msa_dataset_collate(batch):
    good_tokens_list = []
    bad_tokens_list = []
    full_tokens_list = []
    msa_list = []
    name_list = []
    msa_array_list = []
    for good_tokens, bad_tokens, full_tokens, msa, name, msa_array in batch:
        msa_list.append(msa)
        good_tokens_list.append(good_tokens.float())
        bad_tokens_list.append(bad_tokens.float())
        full_tokens_list.append(full_tokens.float())
        name_list.append(name)
        msa_array_list.append(msa_array)
    return good_tokens_list, bad_tokens_list, full_tokens_list, msa_list, name_list, msa_array_list

class Dataset_matches(Dataset):
    def __init__(self, Datapath, data_num='all', matches=True):
        self.Datapath = Datapath
        self.pairs = os.listdir(os.path.join(Datapath, 'K1_K2'))
        self.data_num = data_num
        self.matches = matches
        self.ssn_ratio = 0.8

    def __getitem__(self, idx):
        pairsname = os.path.splitext(self.pairs[idx])[0]
        parts = pairsname.split('-')
        seq = parts[0]
        img1_name = parts[1]
        img2_name = parts[2]
        img1_path = os.path.join(self.Datapath, 'images', seq+'-'+img1_name+'.jpg')
        img2_path = os.path.join(self.Datapath, 'images', seq+'-'+img2_name+'.jpg')
        K1_K2_path = os.path.join(self.Datapath, 'K1_K2', pairsname+'.npy')

        if self.matches:
            matches_path = os.path.join(self.Datapath, 'matches', pairsname+'.npy')
            matches_scores_path = os.path.join(self.Datapath, 'matches_scores', pairsname+'.npy')
            matches = np.load(matches_path, allow_pickle=True)
            matches_scores = np.load(matches_scores_path, allow_pickle=True).reshape((-1))
            desc_path = os.path.join(self.Datapath, 'sift_desc', pairsname+'.npy')
            desc = np.load(desc_path, allow_pickle=True)
            des1 = desc[:, :128]
            des2 = desc[:, 128:]
            pts1 = np.concatenate((matches[:, :2], des1), axis=1)
            pts2 = np.concatenate((matches[:, 2:], des2), axis=1)
            matches = np.concatenate((pts1, pts2), axis=1)
            mask = matches_scores <= self.ssn_ratio
            good_matches = matches[mask,:]
            bad_matches = matches[~mask,:]

        return img1_path, img2_path, pairsname, good_matches, bad_matches

    def __len__(self):
        if self.data_num == 'all':
            return len(self.pairs)
        else:
            return self.data_num

