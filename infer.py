from src.resources import hparams
import torch
from src.resources.utils import *
import json
from src.resources.dataloader import NERData
from torch.utils.data import DataLoader
from model.bilstm_crf import BiLSTM_CRF
from tqdm import tqdm
def infer(datas):
    words2index = json.load(open(hparams.vocab_path,"r"))
    tags2index = json.load(open(hparams.tag2index_path,"r"))
    index2tag = {}
    for tag, index in tags2index.items():
      index2tag[str(index)] = tag

    datas = [words2index[word] if word in words2index else hparams.UNK_TOKEN_ID for word in datas]
    datas = torch.tensor(datas).view(1, -1)

    embeddings = torch.tensor(load_embeddings(), dtype=torch.float).to("cuda")
    model = BiLSTM_CRF(
              embeddings,
              hparams.nb_labels, 
              emb_dim=hparams.embedding_dim, 
              hidden_dim=hparams.hidden_dim, 
              cuda = 'cpu'
            )

    model.load_state_dict(torch.load(hparams.best_checkpoint_path), strict=False)
    _, output_label = model(datas, None)
    print(output_label[0])
    output =[index2tag[str(index)] for index in output_label[0]]
    print(output)


