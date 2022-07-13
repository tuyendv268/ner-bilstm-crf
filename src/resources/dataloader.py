import torch
from torch.utils.data import Dataset
from src.resources import hparams


class NERData(Dataset):
    def __init__(self, X, y, masks):
        self.datas = X
        self.labels = y
        self.masks = masks

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        X = self.datas[index]
        y = self.labels[index]
        mask = self.masks[index]

        return X, y, mask