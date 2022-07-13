from resources import hparams
import json
from resources.utils import load_data

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
