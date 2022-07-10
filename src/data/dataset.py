from torch.utils.data import Dataset
from src.resources import utils
from src.resources import hparams
import torch

class NERData(Dataset):
    def __init__(self, sents, labels):
        self.word2index = utils.load_word2index(hparams.word2index_path)
        self.tag2index = utils.load_tag2index(hparams.tag2index_path)
        
        tmp_sent = utils.convert_word2index(sents, self.word2index, hparams.max_sent_length)
        tmp_label = utils.convert_tag2index(labels, self.tag2index, hparams.max_sent_length)
        
        self.sents = tmp_sent
        self.labels = tmp_label
        self.masks = (self.sents != hparams.PAD_TOKEN_ID)
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        X = self.sents[index]
        y = self.labels[index]
        mask = self.masks[index]

        return {
            "sent": X,
            "label":y,
            "mask":mask
        }