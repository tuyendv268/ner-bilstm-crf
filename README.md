## Named Entity Recognition
#### Model Architecture:
  + Pretrained Word Embedding + BiLSTM + CRF

#### Library:
  + pytorch
  + tqdm
  + numpy
  + gensim
  
#### Data : 
  + VLSP 2016
  
## Kết quả thử nghiệm:  
Entity | precision	 | recall | F1-score
---|---|---|---
`LOC` | 0.88 | 0.91 | 0.89 
`MISC` | 0.94 | 0.94 | 0.94 
`ORG` | 0.75 | 0.58 | 0.65 
`PER` | 0.87 | 0.81 | 0.84
---|---|---|---
`micro avg` | 0.87 | 0.82 | 0.84
`macro avg` | 0.86 | 0.81 | 0.83
`weighted avg` | 0.86 | 0.82 | 0.84