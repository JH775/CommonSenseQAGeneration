import csv
import pickle
import numpy as np
import pandas as pd
import json

import os
import requests

#READ Context from COSMO dataset

X = pd.read_csv('train.csv')

X_noduplicates = X.drop_duplicates(subset=['context'])
print(X.context)
np.savetxt('input_noduplicates.txt', X_noduplicates.context,fmt='%s',delimiter='\t')


#READ DATA FROM datasets class###############################################################

from datasets import list_datasets, load_dataset, list_metrics, load_metric

ds_list = ["cos_e","common_gen","social_i_qa"]


dataset = load_dataset(ds_list[0], 'v1.11')

# Print all the available datasets
print(dataset.column_names)
all_data = dataset['train']['extractive_explanation'] + dataset['validation']['extractive_explanation']
print(all_data[0],len(dataset), type(dataset))
all_data = [x for x in all_data if not '?' in x] #all_data.loc[all_data.str.contains('?') == False]
list_data = [x for x in all_data if len(x.split(" ")) <= 3 ]
all_data = [x for x in all_data if len(x.split(" ")) > 3]
print(list_data)
df = pd.DataFrame (all_data,columns=['context'])
np.savetxt('input_'+ds_list[0]+'.txt', df.context,fmt='%s',delimiter='\t')


## GET DATA FROM https://github.com/sheng-z/
all_data = pd.DataFrame(columns=['CONTEXT','HYPOTHESIS','LABEL','CONTEXT_FROM','HYPOTHESIS_FROM','CONTEXT_ID','HYPOTHESIS_ID','SUBSET'])
for L in ['A','B']:
    for S in [ 'dev', 'test', 'train']:
        dataset = pd.read_csv( L+"/" +L+ "." + S + '.csv')
        all_data = all_data.append(dataset)



all_data = all_data.drop(columns= ['LABEL','CONTEXT_FROM','HYPOTHESIS_FROM','CONTEXT_ID','HYPOTHESIS_ID','SUBSET'])
print(all_data)

# prepare data for nrl model ##########################
with open('/nrlQA-input/input_nrlQA_HYPOTHESIS.txt', 'w') as outfile:
    #json.dump(data, outfile)
    for row in all_data.HYPOTHESIS:
        x = {"sentence": row.strip("\n") } 
        sentence = json.dumps(x)
        outfile.write(sentence+"\n")
outfile.close()

with open('/nrlQA-input/input_nrlQA_CONTEXT.txt', 'w') as outfile:
    #json.dump(data, outfile)
    for row in all_data.CONTEXT:
        x = {"sentence": row.strip("\n") } 
        sentence = json.dumps(x)
        outfile.write(sentence+"\n")
outfile.close()



