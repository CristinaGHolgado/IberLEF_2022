# -*- coding: utf-8 -*-
"""
Created on Sun May  1 18:33:56 2022

@author: Cristina GH
"""
import pandas as pd
import csv
import sys
import time
import tqdm
import os
import platform
from scipy import stats
from collections import Counter
import argparse

def merge_outputs(folder_path, output_folder):
    df_merges = []
    for file in os.listdir(folder_path):
        if '_test' in file:
            if platform.system().lower() == "windows":
                df = pd.read_csv(f"{folder_path}\\{file}", sep='\t')
            else:
                df = pd.read_csv(f"{folder_path}\\{file}", sep='\t')

            del df['Unnamed: 0']
            
            if df.columns[1] in ["gender","profession","ideology_binary","ideology_multiclass"]:
                col = df.columns[1]
                df_new = df.groupby('label').agg({col: lambda x: stats.mode(x)[0][0]})
                df_merges.append(df_new)
    
    merged_outputs = pd.concat([df for df in df_merges], axis=1).reset_index()
    merged_outputs = merged_outputs[['label','gender','profession','ideology_binary','ideology_multiclass']]
    
    name = "_".join(os.listdir(folder_path)[0].split('_')[2:]).replace("_test","_mergedTest")
    merged_outputs.to_csv(f"{output_folder}merged_output_bert_beto.csv", sep=",", index=False)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-input_folder', '--input_folder', required=True,
                    help="Path to input folder")
    parser.add_argument('-output_folder', '--output_folder', required=True,
                    help="Path to output folder to save merged preds")
    
    args = parser.parse_args()
    
    merge_outputs(args.input_folder, args.output_folder)