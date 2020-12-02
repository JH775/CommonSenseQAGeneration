import csv
import pickle
import numpy as np
import pandas as pd

import os
import requests
import json
import re

# process dataset and add Id's 

#sentence \t  question \t  answer \t label \t category 
dev = pd.read_csv('dev_3783.tsv', sep='\t')
test = pd.read_csv('test_9442.tsv', sep='\t')
dev.columns =['sentence', 'question', 'answer', 'label', 'category'] 
test.columns =['sentence', 'question', 'answer', 'label', 'category']
all_data = dev.append(test)
categories = list(all_data.category.unique())

answers = all_data
answers = answers.drop(columns=['sentence','label', 'question']).drop_duplicates()
print(answers)

all_data['id'] = all_data.index.astype(int)
all_data['id'] = all_data['id'].map(str) + '_MC_TACO'
train, val, test = np.split(all_data.sample(frac=1), [int(.6*len(all_data)), int(.8*len(all_data))])
#test['id'] = test.index.astype(int) 
#test['id'] = test['id'].map(str) + '_MC_TACO'

print(dev.columns, dev.dtypes)
print(test.columns,test.dtypes)
test.to_csv('test.csv', index=False) 
train.to_csv('train.csv', index=False) 
val.to_csv('val.csv', index=False) 

#70% train, 15% val, 15% test
#80% train, 10% val, 10% test
#60% train, 20% val, 20% test