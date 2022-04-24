# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:50:32 2022

@author: Cristina GH
"""

import argparse
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
from utils import EarlyStopping
import sklearn
from sklearn.model_selection import train_test_split

from data import load_data, spanish_dataset
from model import BertClassifier, LSTM, BiLSTM_Attention
from testing import load_and_run


def train(model, train_dataloader, val_dataloader, args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= args.LR)
    
    early_stopping = EarlyStopping(patience=5, verbose=True, save_path=f"{args.save_dir}/bertmodel_{datetime.now()}.pth")

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(args.EPOCHS):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader, desc="Training"):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            
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
        model.eval()
        with torch.no_grad():
            
            for val_input, val_label in tqdm(val_dataloader,desc="Validation"):
                
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.to(torch.long))
                total_loss_val += batch_loss.item()
                    
                acc = (output.argmax(dim=1) == val_label.to(torch.long)).sum().item()
                total_acc_val += acc
            
        if args.stop_early:
            early_stopping(total_loss_val, model)
            if early_stopping.early_stop:
                print('Early stopping')
                break 
        
        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(df_train): .3f} \
                | Train Accuracy: {total_acc_train / len(df_train): .3f} \
                | Val Loss: {total_loss_val / len(df_val): .3f} \
                | Val Accuracy: {total_acc_val / len(df_val): .3f}')
     
    if args.save_model : 
        torch.save(model.state_dict(), f"{args.save_dir}/bertmodel_{datetime.now()}.pth"); print('model saved') 

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-train_file', '--train_file', required=True,
                        help="Path to training data") 
    parser.add_argument('-test_file', '--test_file', required=True,
                        help="Path to test data")
    parser.add_argument('-lm', '--lm', default='bert-base-multilingual-cased',
                        help="Hugging face language model name")
    parser.add_argument('-type', '--type', default='bert',
                        help="Model type")
    parser.add_argument('-lclass', '--lclass', default='gender',
                        help="Class label for the classifier")
    parser.add_argument('--save_model', action="store_true", 
						help='whether to save the model')
    parser.add_argument('--save_dir', default="./../../logs",
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
    df_train, df_val = train_test_split(load_data(args.train_file, agg=0), test_size=25)
    df_test = load_data(args.test_file, agg=0)


    
    dtrain, dval = spanish_dataset(df_train, args.lm, args.lclass), spanish_dataset(df_val, args.lm, args.lclass)
    train_dataloader = torch.utils.data.DataLoader(dtrain, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dval, batch_size=args.BATCH_SIZE)
    nclass = len(dtrain.label_encoder)
    
    if args.type == 'bert':
        model = BertClassifier(args.lm, nclass)
    if args.type == 'bert_lstm':
        model = LSTM(args.lm, nclass)
    if args.type == 'bert_lstm_att':
        model = BiLSTM_Attention(args.lm, nclass)

    model = train(model, train_dataloader, val_dataloader, args)

    # testing
    load_and_run(df_test, args, model)

