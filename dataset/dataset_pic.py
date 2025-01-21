import os
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class Imagenet_val(Dataset):
    def __init__(self, data_dir, image_size=256, data_num='all'):
        super(Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.image_size = image_size  
        self.train_fns = sorted(glob.glob(os.path.join(self.data_dir, "*")))
        self.transformer = transforms.Compose([transforms.ToTensor()])
        if data_num != 'all':
            self.train_fns = self.train_fns[:data_num]

    def __getitem__(self, index):
        fn = self.train_fns[index]
        im = cv2.imread(fn)
        H, W = fn.split('_')[-2:]
        W = W.split('.')[0]
        size = (int(H), int(W), 3)

        im = im.astype(np.float32) / 255.0
        im = self.transformer(im)

        return im, size

    def __len__(self):
        return len(self.train_fns)
    

class ValidationDataset(Dataset):
    def __init__(self, dataset_dir, image_size=256):
        super(ValidationDataset, self).__init__()
        self.fns = glob.glob(os.path.join(dataset_dir, "*"))
        self.fns.sort()
        self.image_size = image_size
        self.transformer = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        fn = self.fns[index]
        im = cv2.imread(fn)
        size = im.shape
        im = cv2.resize(im, (self.image_size, self.image_size))
        im = im.astype(np.float32) / 255.0
        im = self.transformer(im)
        return im, size

    def __len__(self):
        return len(self.fns)


class ValidationDatasetBSD300(Dataset):
    def __init__(self, dataset_dir, image_size=256):
        super(ValidationDatasetBSD300, self).__init__()
        self.fns = glob.glob(os.path.join(dataset_dir, "test", "*"))
        self.fns.sort()
        self.image_size = image_size
        self.transformer = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        fn = self.fns[index]
        im = cv2.imread(fn)
        size = im.shape
        im = cv2.resize(im, (self.image_size, self.image_size))
        im = im.astype(np.float32) / 255.0
        im = self.transformer(im)
        return im, size

    def __len__(self):
        return len(self.fns)
