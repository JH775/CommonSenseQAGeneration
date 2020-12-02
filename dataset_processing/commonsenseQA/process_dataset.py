
import pandas as pd
import numpy as np
import json 

#dataset = load_dataset("commonsense_qa")
#train_json = json.loads('./data/train_rand_split.jsonl')
#test_json = json.loads('./data/test_rand_split_np_answers.jsonl')
#dev_json = json.loads('./data/dev_rand_split.jsonl')
#print(dataset.shape,dataset['train'].shape) 
#for k in dataset.keys:
#    print(k)
train = []
dev = []
test = []
with open('./data/train_rand_split.jsonl') as dataFile:
    for line in dataFile.readlines():
        jsonObj = json.loads(line)
        train.append([jsonObj['id'],jsonObj['question']['stem'], jsonObj['question']['choices'][0]['text'],jsonObj['question']['choices'][1]['text'],jsonObj['question']['choices'][2]['text'],jsonObj['question']['choices'][3]['text'],jsonObj['question']['choices'][4]['text'], jsonObj['answerKey']])

df_train = pd.DataFrame(
  train,
    columns=['ID','question', 'A', 'B','C','D','E','label']
) 
with open('./data/dev_rand_split.jsonl') as dataFile:
    for line in dataFile.readlines():
        jsonObj = json.loads(line)
        dev.append([jsonObj['id'],jsonObj['question']['stem'], jsonObj['question']['choices'][0]['text'],jsonObj['question']['choices'][1]['text'],jsonObj['question']['choices'][2]['text'],jsonObj['question']['choices'][3]['text'],jsonObj['question']['choices'][4]['text'], jsonObj['answerKey']])

df_dev = pd.DataFrame(
  dev,
    columns=['ID','question', 'A', 'B','C','D','E','label']
) 

with open('./data/test_rand_split_no_answers.jsonl') as dataFile:
    for line in dataFile.readlines():
        jsonObj = json.loads(line)
        test.append([jsonObj['id'],jsonObj['question']['stem'], jsonObj['question']['choices'][0]['text'],jsonObj['question']['choices'][1]['text'],jsonObj['question']['choices'][2]['text'],jsonObj['question']['choices'][3]['text'],jsonObj['question']['choices'][4]['text'],jsonObj['answerKey']])

df_test = pd.DataFrame(
  test,
    columns=['ID','question', 'A', 'B','C','D','E','label']
) 

print(df_train.columns)
df_train.to_csv('./datasetMC/train_commonsense_qa.csv', index=False)
df_train.to_csv('./datasetMC/test_commonsense_qa.csv', index=False)
df_dev.to_csv('./datasetMC/val_commonsense_qa.csv', index=False)
