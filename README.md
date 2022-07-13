# ner-bilstm-crf
- lib:
  + pytorch
  + tqdm
  + numpy
  + gensim
- data : vlsp 2016:
  + number of sentence : 18060 
  
- Kết quả thử nghiệm:  
  - Tập val (sau 50 epoch): F1 = tensor(0.9688)
    + '<pad>': tensor(nan), 
    + '<s>': tensor(0.6666), 
    + '</s>': tensor(0.9997), 
    + 'O': tensor(0.9952), 
    + 'B-LOC': tensor(0.8889), 
    + 'B-ORG': tensor(0.6889), 
    + 'I-LOC': tensor(0.8656), 
    + 'B-PER': tensor(0.8277), 
    + 'I-ORG': tensor(0.7631), 
    + 'B-MISC': tensor(0.9841), 
    + 'I-MISC': tensor(0.9333)}
  - Tập test:
    +