embedding_dim=100
batch_size=64 
max_sent_length = 64
nb_labels=11
hidden_dim=256
max_epoch = 1
dropout=0.1
lr = 0.0001

result_path="results/f1_%EPOCH%.txt"
best_checkpoint_path = "checkpoint/checkpoint30.pt"
checkpoint_path="checkpoint/checkpoint_%EPOCH%.pt"

word2index_path="src/resources/vocab/word2index.json"
tag2index_path="src/resources/vocab/tag2index.json"

embedding_path = "src/resources/pretrained/embeddings_wiki_100.model"

train_path="src/resources/data/ner_train_phonlp.txt"
valid_path="src/resources/data/ner_valid_phonlp.txt"
test_path="src/resources/data/ner_test_phonlp.txt"

# padding = 0
# unk = 1
# eos = 2
# bos = 3
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 0
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 1
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3

PAD_TAG, PAD_TAG_ID = "<pad>", 0
BOS_TAG, BOS_TAG_ID = "<s>", 1
EOS_TAG, EOS_TAG_ID = "</s>", 2

