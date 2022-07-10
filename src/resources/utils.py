import torch
from src.resources import hparams
import json
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec 
 

def load_data(path):
    print("----------------Loading data----------------")
    f = open(path, "r",encoding="utf-8")
    lines = f.readlines()

    datas = []
    datas_temp = []
    labels = []
    labels_temp =[]
    for line in tqdm(lines):
        temp = line.replace("\n","").strip().split("\t")
        if(line == "\n"):
            datas.append(datas_temp)
            labels.append(labels_temp)
            datas_temp = []
            labels_temp = []
        else:
            datas_temp.append(temp[0].replace("_"," "))
            labels_temp.append(temp[1])
    print("----------------Load successful----------------")
    return datas, labels

def load_embeddings(path):
    print("loading pretrained embedding from: ", path)
    embeddings = Word2Vec.load(path).wv.vectors

    pad = np.array([[1]*hparams.embedding_dim])
    bos = np.random.randn(1, hparams.embedding_dim)
    eos = np.random.randn(1, hparams.embedding_dim)
    unk = np.random.randn(1, hparams.embedding_dim)
    
    embeddings = np.concatenate((pad,bos,eos,unk,embeddings), axis=0)
    print("load succesfull!")
    return embeddings

def build_word2index_dict(embedding_path):
    tmp = Word2Vec.load(embedding_path).wv
    tmp_list = tmp.index2word
    word2index = {}
    word2index[hparams.PAD_TOKEN] = hparams.PAD_TOKEN_ID
    word2index[hparams.BOS_TOKEN] = hparams.BOS_TOKEN_ID
    word2index[hparams.EOS_TOKEN] = hparams.EOS_TOKEN_ID
    word2index[hparams.UNK_TOKEN] = hparams.UNK_TOKEN_ID
    for word in tmp_list:
        if word not in word2index:
            word2index[word] = len(word2index)
    with open(hparams.vocab_path, "w") as tmp:
        tmp.write(json.dumps(word2index, ensure_ascii=False))
    return word2index

def build_tags2index_dict(labels):
    tag2idx = {
        hparams.PAD_TAG: hparams.PAD_TAG_ID,
        hparams.BOS_TAG: hparams.BOS_TAG_ID,
        hparams.EOS_TAG: hparams.EOS_TAG_ID,
    }
    for tags in labels:
        for  tag in tags:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    with open(hparams.tag2index_path, "w") as tmp:
        tmp.write(json.dumps(tag2idx, ensure_ascii=False))
    return tag2idx

def load_word2index(path):
    word2index = json.load(open(path, "r", encoding="utf-8"))
    return word2index

def load_tag2index(path):
    tag2index = json.load(open(path, "r", encoding="utf-8"))
    return tag2index

def convert_word2index(datas, word2index, max_sent_length):
    datas = [sent[0:max_sent_length-2] for sent in datas]
    datas = [[hparams.BOS_TOKEN] + line + [hparams.EOS_TOKEN] + [hparams.PAD_TOKEN]*(max_sent_length-len(line)) for line in datas]
    datas = [[word2index[word] if word in word2index else hparams.UNK_TOKEN_ID for word in line] for line in datas]

    return torch.tensor(datas)

def convert_tag2index(tags, tags2index, max_sent_length):
    tags = [sent[0:max_sent_length-2] for sent in tags]
    tags = [[hparams.BOS_TAG]+line +[hparams.EOS_TAG]+ [hparams.PAD_TAG]*(max_sent_length-len(line)) for line in tags]
    tags = [[tags2index[tag] for tag in line] for line in tags]

    return torch.tensor(tags)

def convert_index2tag(index):
    index2tag = {}
    path = hparams.tag2index_path
    tags2index = json.load(open(path,"r"))
    for keys, values in tags2index.items():
        index2tag[str(values)] = keys
    
    index = [[index2tag[str(tag)] for tag in tags] for tags in index]
    return index



def build_vocab():
    train_data, _ = load_data(hparams.training_path)
    val_data, _ = load_data(hparams.val_path)
    test_data, _=load_data(hparams.test_path)

    vocab = train_data+val_data+test_data

    word_to_ix = {
        hparams.UNK_TOKEN: hparams.UNK_TOKEN_ID,
        hparams.PAD_TOKEN: hparams.PAD_TOKEN_ID,
        hparams.BOS_TOKEN: hparams.BOS_TOKEN_ID,
        hparams.EOS_TOKEN: hparams.EOS_TOKEN_ID,
    }
    for sentence in vocab:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    json_obj = json.dumps(word_to_ix, ensure_ascii=False)
    with open(hparams.vocab_path,"w") as tmp:
        tmp.write(json_obj)
