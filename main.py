from src.resources import hparams
from src.model.bilstm_crf import BiLSTM_CRF
from src.resources.utils import *
from src.data.dataset import NERData
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from src.trainer import Trainer

cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

train_path = hparams.train_path
val_path = hparams.valid_path
test_path = hparams.test_path

train_datas, train_labels = load_data(path=train_path)
print(f"Total training sample: {len(train_datas)}")

val_datas, val_labels = load_data(path=val_path)
print(f"Total valid sample: {len(val_datas)}")


train_data = NERData(train_datas[0:148], train_labels[0:148])
val_data = NERData(val_datas, val_labels)

train_dl = DataLoader(dataset=train_data, batch_size=hparams.batch_size, shuffle=True)
val_dl = DataLoader(dataset=val_data, batch_size=hparams.batch_size, shuffle=True)


embeddings = torch.tensor(load_embeddings(hparams.embedding_path), dtype=torch.float).to(cuda)
model = BiLSTM_CRF(embeddings, hparams.nb_labels, cuda=cuda, emb_dim=hparams.embedding_dim, hidden_dim=hparams.hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=hparams.lr, weight_decay=1e-5)

trainer = Trainer(model=model, cuda=cuda, optimizer=optimizer, train_dl=train_dl, val_dl=val_dl, mode="train",max_epoch=hparams.max_epoch)

trainer.train()
