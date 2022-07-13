from torch import nn
from src.resources import hparams
from model.bilstm import BiLSTM
from model.crf import CRF

# or try the vectorized version:
# from crf_vectorized import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, embeddings, nb_labels, cuda, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.cuda = cuda
        self.lstm = BiLSTM(
            embeddings, nb_labels, emb_dim=emb_dim, hidden_dim=hidden_dim, cuda=cuda
        ).to(cuda)
        self.crf = CRF(
            nb_labels,
            bos_tag_id=hparams.BOS_TOKEN_ID,
            eos_tag_id=hparams.EOS_TOKEN_ID,
            pad_tag_id=hparams.PAD_TOKEN_ID,
            cuda=self.cuda,
            batch_first=True,
        ).to(cuda)

    def forward(self, x, mask=None):
        emissions = self.lstm(x)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path

    def loss(self, x, y, mask=None):
        emissions = self.lstm(x)
        nll = self.crf(emissions, y, mask=mask)
        return nll