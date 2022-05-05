# textcnn.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score


from utils import EarlyStopping
from data import load_data, spanish_dataset
from testing import load_and_run



class CNNBert(nn.Module):
    
    def __init__(self, lm, lclass, embed_size):
        super(CNNBert, self).__init__()
        filter_sizes = [1,2,3,4,5]
        num_filters = 32
        self.convs1 = nn.ModuleList([nn.Conv2d(4, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, lclass)
        self.sigmoid = nn.Sigmoid()
        self.bert_model = AutoModel.from_pretrained(lm, output_hidden_states=True)

    def forward(self, x, input_masks, token_type_ids):
        x = self.bert_model(x, attention_mask=input_masks, token_type_ids=token_type_ids)[1][-4:]#; print(x)#
        x = torch.stack(x, dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)
        return self.sigmoid(logit)


def train(model, trainloader, valdataloader, criterion, args, device):
       
    #optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=0.9)
    
    #if num:
        #writer = SummaryWriter(f'runs/gru_experiment_{num}')

    warmup = 0
    
    trlosses = []
    vlosses = []
    epoch_times  = []
    p = 0
    
    for epoch in range(args.EPOCHS):
        model.train()
        start = time.time()
        avg_loss = 0
        counter = 0
        
        for x, y in tqdm(trainloader, desc='Training'):
            #print(x, y)
            counter +=1
            p +=1
            ids = x['input_ids'].squeeze(1).to(device)
            masks = x['attention_mask'].squeeze(1).to(device)
            type_ids = x['token_type_ids'].squeeze(1).to(device)
            #print(ids.shape, masks.shape, type_ids.shape)
            optimizer.zero_grad()
            #model.zero_grad()
            out = model(x = ids, input_masks=masks, token_type_ids =type_ids)
            del ids
            del masks
            del type_ids
            gc.collect()
            torch.cuda.empty_cache()

            loss = criterion(out, y.to(device))
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            avg_loss += loss.item()
            
            
            #if counter % 200 == 0:
             #   print(f"Epoch : {epoch}, Step: {counter}/{len(trainloader)} ==> Avg Loss for epoch: {avg_loss/counter}")
                #if num: writer.add_scalar('training_loss', avg_loss/counter, p)
        print(f"epoch {epoch+1}, training loss {avg_loss}") 
        trlosses.append(avg_loss)
        model.eval()
        o,t, _ = evaluate(model, valdataloader,criterion, params, device)
        vlosses.append(_)
        if epoch > warmup:
            early_stopping(_[0], model)
            if early_stopping.early_stop:
                print('Early stopping')
                break 
                
        current_time = time.time()
        #print(f"Total time elapsed: {current_time-start} seconds")
        epoch_times.append(current_time- start)
    
    #print(f"Total Training Time: {sum(epoch_times)} seconds")
    return model, trlosses, vlosses


def evaluate(model, testdataloader, criterion, params, device):
    
    device = device
    
    model.eval()
    outputs = []
    targets = []
    losses = []
    start = time.time()
    
    for x,y in tqdm(testdataloader, desc = 'Evaluating'):
        ids = x['input_ids'].squeeze(1).to(device)
        masks = x['attention_mask'].squeeze(1).to(device)
        type_ids = x['token_type_ids'].squeeze(1).to(device)
        with torch.no_grad():
            out = model(ids, masks,type_ids)
            
        del ids
        del masks
        del type_ids
        gc.collect()
        torch.cuda.empty_cache()
        #out = torch.squeeze(out, dim=0)
        #print(out.shape,out.argmax(dim=1))
        loss = criterion(out, y.to(device))
        losses.append(loss.item())
        outputs.extend(out.argmax(dim=1).cpu().detach().numpy())
        targets.extend(y.numpy())
    
    score = f1_score(targets, outputs, average='micro')
    print(f'F1 score : ', score)
    return np.asarray(outputs), np.asarray(targets), losses



if __name__ == '__main__':


	parser = argparse.ArgumentParser()
    
    parser.add_argument('-train_file', '--train_file', required=True,
                        help="Path to training data")
    parser.add_argument('--augment', '--augment', action="store_true", 
                        help="Path to augmented training data")
    parser.add_argument('-val_file', '--val_file', default=None,
                        help="Path to val data")
    parser.add_argument('-test_file1', '--test_file1', required=True,
                        help="Path to test data")
    parser.add_argument('-test_file2', '--test_file2', required=True,
                        help="Path to test data unlabeled data")
    parser.add_argument('-lm', '--lm', default='bert', choices = ['mbert', 'beto', 'xlm17', 'xlm100', 'xlmrb', 'xlmrl'],
                        help="Hugging face language model name")
    parser.add_argument('-type', '--type', default='cnnbert', choices = ['cnnbert'],
                        help="Model type")
    parser.add_argument('-lclass', '--lclass', default='gender',
                        help="Class label for the classifier")
    parser.add_argument('--save_model', action="store_true", 
						help='whether to save the model')
    parser.add_argument('--save_dir', default="~/IberLEF_2022/logs",
                     help='Model dir')
    parser.add_argument('--EPOCHS', default=2, type=int,
                     help='Number of epochs. Default 2')
    parser.add_argument('--LR', default=1e-6, type=int,
                     help='Learning rate')
    parser.add_argument('--BATCH_SIZE', default=2, type=int,
                     help='Batch size')
    parser.add_argument('--stop_early', action="store_true", 
						help='whether to use early stopping')

    
    args = parser.parse_args()
    
    np.random.seed(112)
    # language model dictionary
    lmdict = { 'xlm17' : 'xlm-mlm-17-1280',
              'xlm100' : 'xlm-mlm-100-1280',
              'beto' : 'dccuchile/bert-base-spanish-wwm-cased',
               'mbert' : 'bert-base-multilingual-cased',
               'xlmrb' : 'xlm-roberta-base',
                 'xlmrl' : 'xlm-roberta-large'}
    args.lm = lmdict[args.lm]

    #if args.val_file is None:
    df_train = load_data(args.train_file, agg=0)
    df_val = load_data(args.val_file, agg=0)
    if args.augment :
        print(f'Going to use augment for {args.lclass}')
        aug_data = pd.read_csv(f'../../notebook/augment_{args.lclass}.csv', sep='\t')
        #print(aug_data.head())
        print('==>\n', df_train[['tweet',args.lclass]])
        #print('===>\n', aug_data[['tweet',args.lclass]])
        added_df = pd.concat([df_train[['tweet',args.lclass]],
                                  aug_data[['tweet',args.lclass]]], axis=0).reset_index()
        #print(added_df)
        ldict = dict()
        for l in added_df[args.lclass]:
            if l in ldict:
                ldict[l] += 1
            else:
                ldict[l] = 1

        print("New traindata label distribution :", ldict)
        df_train = added_df

    
    # validation data provided by organizers
    df_test1 = load_data(args.test_file1, agg=0)
    print("DataSplit:", len(df_train), len(df_val), len(df_test1))
    
    dtrain, dval = spanish_dataset(df_train, args.lm, args.lclass), spanish_dataset(df_val, args.lm, args.lclass)
    train_dataloader = torch.utils.data.DataLoader(dtrain, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dval, batch_size=args.BATCH_SIZE)
    nclass = len(dtrain.label_encoder)

    criterion = nn.CrossEntropyLoss()

    model = CNNBert(args.lm, nclass, 768)

    gc.collect()
	torch.cuda.empty_cache()
	early_stopping = EarlyStopping(patience=10, verbose=True, save_path=f"{args.save_dir}biobert_pubmed_cnn.pth")

	net = CNNBert(args.lm, 3, 768).to(device)

	net, trl, vl = train(net, train_dataloader, val_dataloader,  criterion, args, device)


	# testing
    #print(dtrain.label_encoder)
    #print(df_test)
    print("\nTesting on real development set")
    load_and_run(df_test1, args, dtrain.label_encoder, model, name="devset")
    print('#'*20)
    print("*"*20,"Getting output on unseen test", "*"*20)
    df_test2 = load_data(args.test_file2, agg=0)
    load_and_run(df_test2, args, dtrain.label_encoder, model, no_labels=True, name="test")
    '''