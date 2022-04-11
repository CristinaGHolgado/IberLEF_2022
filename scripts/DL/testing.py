import torch
from torch import nn

from tqdm import tqdm
import pandas as pd
import argparse

from sklearn.metrics import classification_report

from model import BertClassifier
from data import spanish_dataset

def load_and_run(data, args):
    # model path
    path = '../../logs/'

    #loading model
    model = BertClassifier()
    model.load_state_dict(torch.load(f"{path}{args.modelname}"))
    #print(model)
    model.eval()
    print('Model loaded ...')

    val = spanish_dataset(data)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=100)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()

    total_acc_val = 0
    total_loss_val = 0
    preds, truth = [], []
    for val_input, val_label in tqdm(val_dataloader):
                
        val_label = val_label.to(device)
        mask = val_input['attention_mask'].to(device)
        input_id = val_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        batch_loss = criterion(output, val_label.to(torch.long))
        total_loss_val += batch_loss.item()
        
        truth.extend(val_label.detach().cpu().numpy())
        preds.extend(output.argmax(dim=1).detach().cpu().numpy())

        acc = (output.argmax(dim=1) == val_label.to(torch.long)).sum().item()
        total_acc_val += acc
    
    class_report = classification_report(preds, 
                                              truth, 
                                              zero_division = 0, 
                                              digits = 6, 
                                              )
    print(class_report)        
    print(f' Val Loss: {total_loss_val / len(data): .3f} \
                | Val Accuracy: {total_acc_val / len(data): .3f}')




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-modelname', '--modelname', required=True,
                        help="name of model") 
    parser.add_argument('-filepath', '--filepath', required=True,
                        help = "file path of testing file")

    args = parser.parse_args()
    
    # read the testing data
    df = pd.read_csv(args.filepath)
    df = df.dropna()
    df['tweet'] = df['tweet'].str.replace('@user','')
    df = df[df.tweet.str.len() > 10]


    load_and_run(df, args)

    # save the models in the logs folder 
    # python testing.py --modelname 'bertmodel_2022-04-11 16:14:52.257235.pth'  --file ../../data/development_test.csv 
