import torch
from torch import nn

from tqdm import tqdm
import pandas as pd
import argparse
import gc

from sklearn.metrics import classification_report

from model import BertClassifier
from data import spanish_dataset

def load_and_run(df, args, label_dict, model=None, no_labels=False, name='val'):
    # model path
    path = '../../logs/'
    id2label = dict(zip(label_dict.values(), label_dict.keys()))
    # when no labels
    # add false labels
    if no_labels:
        df[args.lclass] = 0
        print('adding false labels for testing_no_labels files...')

    val = spanish_dataset(df, args.lm, args.lclass)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=args.BATCH_SIZE)


    if model is None:
        #loading model
        model = BertClassifier()
        model.load_state_dict(torch.load(f"{path}{args.modelname}"))
        print('Model weights loaded ...')
    else:
        print("Using current trained model...")

    #print(model)
    
    model.eval()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()

    total_acc_val = 0
    total_loss_val = 0
    preds, truth = [], []
    with torch.no_grad():
        for val_input, val_label in tqdm(val_dataloader, desc='Evaluating'):
            
            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            if 'xlm-m' in args.lm:
                mask = mask.squeeze(1)
            if args.type ==  'cnnbert':
                type_ids = val_input['token_type_ids'].squeeze(1).to(device)

                output = model(input_id, mask, type_ids)
                del input_id
                del mask
                del type_ids
                
            else:
                output = model(input_id, mask)
                del input_id
                del mask
                
            
            gc.collect()
            torch.cuda.empty_cache()

            batch_loss = criterion(output, val_label.to(torch.long))
            total_loss_val += batch_loss.item()
            
            truth.extend(val_label.detach().cpu().numpy())
            preds.extend(output.argmax(dim=1).detach().cpu().numpy())

            acc = (output.argmax(dim=1) == val_label.to(torch.long)).sum().item()
            total_acc_val += acc
        
        class_report = classification_report(truth, 
                                                  preds, 
                                                  zero_division = 0, 
                                                  digits = 6, 
                                                  )
        print(class_report)        
        print(f' Val Loss: {total_loss_val / len(df): .3f} \
                    | Val Accuracy: {total_acc_val / len(df): .3f}')

    if args.augment:
        name = name+'aug'
    
    # create output file
    #preds = [0] *  len(df)
    output_df = pd.DataFrame()
    output_df['label'] = df.label
    output_df[args.lclass] =  [id2label[l] for l in preds]
    output_df.to_csv(f'output_{args.lclass}_{args.type}_{args.lm[:4]}_{name}.csv', sep='\t')

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

