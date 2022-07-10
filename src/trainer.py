from tqdm import tqdm
from seqeval.metrics import classification_report
from src.resources import hparams
from src.resources import utils
import numpy as np
import torch

class Trainer():
    def __init__(self, model, cuda, optimizer, train_dl, val_dl, mode, max_epoch):
        self.model=model
        self.cuda = cuda
        self.optimizer=optimizer
        self.train_dl=train_dl
        self.val_dl=val_dl
        self.mode=mode
        self.max_epoch=max_epoch
    
    def train(self):
        train_losses=[]
        val_losses =[]
        for epoch in range(self.max_epoch):
            tqdm_dl = tqdm(self.train_dl)
            loss_list = []
            for data_sample in tqdm_dl:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = data_sample["sent"]
                targets = data_sample["label"]
                mask = data_sample["mask"]

                # Step 3. Run our forward pass.
                loss = -self.model.loss(sentence_in, targets, mask=mask)
                loss_list.append(loss)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                self.optimizer.step()
                tqdm_dl.set_postfix({"loss":torch.tensor(loss_list).mean(), "epoch":f"{epoch}/30"})
            train_losses.append(torch.tensor(loss_list).mean())
            result, val_loss = self.val()
            val_losses.append(val_loss)
            
            result_path = hparams.result_path.replace("%EPOCH%", str(epoch))
            with open(result_path, "w") as tmp:
                tmp.write(result)
                print("saved: ", result_path)
            path = hparams.checkpoint_path.replace("%EPOCH%", str(epoch))
            torch.save(self.model.state_dict(), path)
            print("saved: ", path)
        np.save("results/loss/training_loss.npy", np.array(train_losses))
        print("saved training loss.")
        np.save("results/loss/val_loss.npy", np.array(val_losses))
        print("saved valid loss.")
        
    def ignore_label(self, predicts, labels, ignore_label):
        predicts = [[pred for pred, lbl in zip(sent_pred, sent_label) if lbl not in ignore_label] for sent_pred, sent_label in zip(predicts, labels)]
        labels = [[token for token in sent_label if token not in ignore_label] for sent_label in labels]

        return predicts, labels

    
    def remove_padding(self, labels, padding):
        labels = [label[0:pad.sum(dim=0)] for label, pad in zip(labels, padding)]
        return labels
    
    def val(self):
        val_losses =[]
        tqdm_dl = tqdm(self.val_dl)
        predicts = []
        labels = []
        for data_sample in tqdm_dl:
            sentence_in = data_sample["sent"]
            targets = data_sample["label"]
            mask = data_sample["mask"]

            loss = -self.model.loss(sentence_in, targets, mask=mask)
            val_losses.append(loss)
            predict = self.model(sentence_in, mask)
            
            label = self.remove_padding(targets.tolist(), mask)
            labels += label
            predicts += predict
        
        predicts, labels = self.ignore_label(predicts=predicts, labels=labels, ignore_label=[0,1,2])   
        predicts = utils.convert_index2tag(predicts)
        labels = utils.convert_index2tag(labels)
        
        results = classification_report(labels,predicts)
        print(results)
        return results, torch.tensor(val_losses).mean().item()
            
