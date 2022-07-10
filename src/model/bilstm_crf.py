from torch import nn
from src.resources import hparams
from src.model.lstm import BiLSTM
from torchcrf import CRF


# or try the vectorized version:
# from crf_vectorized import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, embeddings, nb_labels, cuda, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.cuda = cuda
        self.lstm = BiLSTM(
            embeddings, cuda, nb_labels, emb_dim=emb_dim, 
            hidden_dim=hidden_dim
        ).to(cuda)
        self.crf = CRF(num_tags=nb_labels,
                       batch_first=True).to(cuda)

    def forward(self, x, mask=None):
        emissions = self.lstm(x)
        path = self.crf.decode(emissions, mask=mask)
        return path

    def loss(self, x, y, mask=None):
        emissions = self.lstm(x)
        nll = self.crf(emissions, y, mask=mask)
        return nll