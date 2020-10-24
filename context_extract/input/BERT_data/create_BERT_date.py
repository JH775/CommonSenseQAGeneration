import csv
import pickle
import numpy as np
import pandas as pd

import os
import requests
import json
import re


## CREATE VOCAB.txt for Bert model

error_file = open('error_file.txt','w')
vocab_list = []
with open('../../generated_questions/10-23.out', 'r') as infile:
    prediction= "start"
    while prediction:
        try:
            infile.readline()
            prediction = infile.readline()
            prediction = prediction.strip("prediction:").replace("{\'", "{\"").replace("\'}", "\"}").replace(", \'", ", \"").replace("\',", "\",")
            prediction = prediction.replace(" \" ", " \' ")
            prediction = json.loads(prediction)
            words_list = prediction['words']
            for word in words_list:
                if len(word) > 3:
                    vocab_list.append(word.lower())
                    
        except ValueError:
            error_file.write("ERROR"+ prediction)
        finally:
            infile.readline()

print(len(vocab_list))
with open ('../input_ALL.txt','r') as vocab_source:
    for line in vocab_source:
        data = line.strip('\t').strip('\n').split(' ')
        for word in data:
            if len(word) > 3:
                vocab_list.append(word.lower())
    
print(len(vocab_list))

df = pd.DataFrame(vocab_list)
print(df.shape)
df = df.drop_duplicates()
print(df.shape)
np.savetxt(r'vocab.txt', df.values, fmt='%s')

## create input  text file from QA_set.csv
"""
csv_QA = pd.read_csv("../QA_set.csv").drop(columns=['start','end'])
np.savetxt('input_bert.txt', csv_QA.values, delimiter='\t', fmt='%s')
"""