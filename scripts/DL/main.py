# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:50:32 2022

@author: Cristina GH
"""
import time
import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
from utils import EarlyStopping
import sklearn
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from data import load_data, spanish_dataset
from model import BertClassifier, LSTM, BiLSTM_Attention, SimpleBert
from testing import load_and_run


def train(model, train_dataloader, val_dataloader, args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    class_weights = torch.tensor(compute_class_weight(class_weight = 'balanced', classes = np.unique(df_train[args.lclass]), y =  df_train[args.lclass]))
    criterion = nn.CrossEntropyLoss(weight=class_weights.float()).to(device)

    # criterion = nn.CrossEntropyLoss() #.to(device)
    optimizer = Adam(model.parameters(), lr= args.LR)
    warmup = 10
    early_stopping = EarlyStopping(patience=5, verbose=True, save_path=f"{args.save_dir}/{args.type}_{args.lclass}_{args.lm[:4]}_model_{datetime.now()}.pth")

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    for epoch_num in range(args.EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        t1 = time.time()
        model.train()
        for train_input, train_label in tqdm(train_dataloader, desc="Training"):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            if 'xlm-m' in args.lm:
                mask = mask.squeeze(1)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            
            #output = model(input_ids = input_id, attention_mask=mask)#, mask)
            output = model(input_id, mask)
                
            batch_loss = criterion(output, train_label.to(torch.long))
            total_loss_train += batch_loss.item()
                
            acc = (output.argmax(dim=1) == train_label.to(torch.long)).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        val_labels, val_preds = [], []
        model.eval()
        with torch.no_grad():
            
            for val_input, val_label in tqdm(val_dataloader,desc="Validation"):
                
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                if 'xlm-m' in args.lm:
                    mask = mask.squeeze(1)
                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.to(torch.long))
                total_loss_val += batch_loss.item()
                val_labels.extend(val_label.detach().cpu().numpy()); val_preds.extend(output.argmax(dim=1).detach().cpu().numpy())
                acc = (output.argmax(dim=1) == val_label.to(torch.long)).sum().item()
                total_acc_val += acc
            
        if epoch_num > warmup and args.stop_early:
            early_stopping(total_loss_val, model)
            if early_stopping.early_stop:
                print('Early stopping')
                break 
        t2 = time.time()
        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train} \
                | Train Accuracy: {total_acc_train / len(df_train): .4f} \
                | Val Loss: {total_loss_val / len(df_val): .6f} \
                | Val Accuracy: {total_acc_val / len(df_val): .4f}\
                | Time : {t2-t1: .4f}')
        print("val confusion matrix", confusion_matrix(val_labels, val_preds))
     
    if args.save_model : 
        torch.save(model.state_dict(), f"{args.save_dir}/final_{args.type}_{args.lclass}_{args.lm[:4]}_model_{datetime.now()}.pth"); print('model saved') 

    return model


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
    parser.add_argument('-lm', '--lm', default='mbert', choices = ['mbert', 'beto', 'xlm17', 'xlm100', 'xlmrb', 'xlmrl'],
                        help="Hugging face language model name")
    parser.add_argument('-type', '--type', default='bert', choices = ['bert', 'bert_linear', 'bert_lstm', 'bert_lstm_att'],
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
        try:
            aug_data = pd.read_csv(f'~/IberLEF_2022/notebook/augment_{args.lclass}.csv', sep='\t')
        except FileNotFoundError:
            aug_data = pd.read_csv(f'data\\augmented\\augment_{args.lclass}.csv', sep='\t')
        print(f"Size of augmented data: {len(aug_data)}")
        print(f"Size of training data: {len(df_train)}")
        #print(aug_data.head())
        print('==>\n', df_train[['tweet',args.lclass]])
        #print('===>\n', aug_data[['tweet',args.lclass]])
        added_df = pd.concat([df_train[['tweet',args.lclass]],
                                  aug_data[['tweet',args.lclass]]], axis=0).reset_index()
        print(f"Size of added augmented data: {len(added_df)}")
        #print(added_df)
        ldict = dict()
        for l in added_df[args.lclass]:
            if l in ldict:
                ldict[l] += 1
            else:
                ldict[l] = 1

        print("New traindata label districution :", ldict)
        df_train = added_df

    #else:
    #    df_val = load_data(args.val_file, agg=0)
    
    # validation data provided by organizers
    df_test1 = load_data(args.test_file1, agg=0)
    print("DataSplit:", len(df_train), len(df_val), len(df_test1))
    
    dtrain, dval = spanish_dataset(df_train, args.lm, args.lclass), spanish_dataset(df_val, args.lm, args.lclass)
    train_dataloader = torch.utils.data.DataLoader(dtrain, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dval, batch_size=args.BATCH_SIZE)
    nclass = len(dtrain.label_encoder)
    
    if args.type == 'bert':
        model = SimpleBert(args.lm, nclass)
    if args.type == 'bert_linear':
        model = BertClassifier(args.lm, nclass)
    if args.type == 'bert_lstm':
        model = LSTM(args.lm, nclass)
    if args.type == 'bert_lstm_att':
        model = BiLSTM_Attention(args.lm, nclass)

    print(f"\n Experiment Info : {args}\n")
    model = train(model, train_dataloader, val_dataloader, args)

    # testing
    #print(dtrain.label_encoder)
    #print(df_test)
    print("\nTesting on real development set")
    load_and_run(df_test1, args, dtrain.label_encoder, model, name="devset")
    print('#'*20)
    print("*"*20,"Getting output on unseen test", "*"*20)
    df_test2 = load_data(args.test_file2, agg=0)
    load_and_run(df_test2, args, dtrain.label_encoder, model, no_labels=True, name="test")
