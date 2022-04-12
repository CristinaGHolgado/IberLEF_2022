# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:50:32 2022

@author: Cristina GH
"""
from data import split_data, spanish_dataset
from model import BertClassifier
import argparse
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
from utils import EarlyStopping


def train(model, train_data, val_data, learning_rate, epochs):
    
    train, val = spanish_dataset(train_data),spanish_dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)
    
    early_stopping = EarlyStopping(patience=5, verbose=True, save_path=f"{args.save_dir}/model_{datetime.now()}.pth")

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
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

        with torch.no_grad():
            
            for val_input, val_label in val_dataloader:
                
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
        
        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
     
    if args.save_model : 
        torch.save(model.state_dict(), f"{args.save_dir}/bertmodel_{datetime.now()}.pth"); print('model saved') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-train_file', '--train_file', required=True,
                        help="Path to training data") 
    parser.add_argument('-test_file', '--test_file', required=True,
                        help="Path to test data")
    parser.add_argument('--save_model', action="store_true", 
						help='whether to save the model')
    parser.add_argument('--save_dir', default="./../../logs",
                     help='Model dir')
    parser.add_argument('--EPOCHS', default=2, type=int,
                     help='Number of epochs. Default 2')
    parser.add_argument('--LR', default=1e-6, type=int,
                     help='Learning rate')
    parser.add_argument('--stop_early', action="store_true", 
						help='whether to use early stopping')
    
    args = parser.parse_args() 

    df_train, df_val, df_test = split_data(args.train_file, args.test_file)._split()
    model = BertClassifier()
    train(model, df_train, df_val, args.LR, args.EPOCHS)
