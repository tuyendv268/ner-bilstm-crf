from src.resources import hparams
from torchmetrics.functional import f1_score

import torch
from src.resources.utils import *
from model.bilstm_crf import BiLSTM_CRF
from src.resources.dataloader import NERData
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report
from tqdm import tqdm
import numpy as np

def validate(model, cuda):
    # embeddings=torch.tensor(load_embeddings(), dtype=torch.float)
    # model = BiLSTM_CRF(embeddings ,hparams.nb_labels, emb_dim=hparams.embedding_dim, hidden_dim=hparams.hidden_dim, cuda = 'cpu')
    # device = torch.device('cpu')
    # model.load_state_dict(torch.load(PATH, map_location=device), strict=False)
    datas, labels = load_data(path=hparams.val_path)
    
    print(f"Total data sample: {len(datas)}")

    words2index = json.load(open(hparams.vocab_path,"r"))
    tags2index = json.load(open(hparams.tag2index_path,"r"))

    val_datas = convert_word2index(datas, words2index,hparams.max_sent_length).to(cuda)
    val_labels = convert_tag2index(labels, tags2index,hparams.max_sent_length).to(cuda)
    masks = (val_datas != hparams.PAD_TOKEN_ID).float()
    # 1984
    traning_datas = NERData(val_datas[0:], val_labels[0:], masks[0:])
    dataloader = DataLoader(traning_datas, batch_size=64, shuffle=True)
    tqdm_dl = tqdm(dataloader)

    val_predict = None
    val_label = None    
    losses = []

    for data, label, mask in tqdm_dl:
        _, predicts = model(data, None)
        loss = model.loss(data, label,mask = mask)
        losses.append(loss)
        predicts = [sent + [tags2index["<pad>"]]*(label.shape[1]-len(sent)) for sent in predicts]
        predicts = torch.tensor(predicts).to(cuda)
        if val_predict == None and val_label == None:
            val_predict = predicts.view(-1)
            val_label = label.view(-1)
        else:
            val_predict = torch.concat((val_predict, predicts.view(-1)), dim=0).to(cuda)
            val_label = torch.concat((val_label, label.view(-1)), dim=0).to(cuda)
    idx2tag = {}
    for tag, idx in tags2index.items():
        idx2tag[str(idx)] = tag
    
    val_predict = [idx2tag[tag] for tag in val_predict]
    val_label = [idx2tag[tag] for tag in val_label]
    
    results = classification_report(labels,predicts)
    
    return results, torch.tensor(losses).mean().item()