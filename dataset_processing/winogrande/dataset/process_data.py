
import pandas as pd
import numpy as np
import json 

train = []
dev = []
test = []
# process winogrande dataset into csv files
with open('./train_l.jsonl') as dataFile:
    for line in dataFile.readlines():
        jsonObj = json.loads(line)
        train.append([jsonObj['qID'],jsonObj['sentence'], jsonObj['option1'],jsonObj['option2'],jsonObj['answer']])
df_train = pd.DataFrame(
  train,
    columns=['ID','question', 'option1', 'option2','label']
) 
with open('./dev.jsonl') as dataFile:
    for line in dataFile.readlines():
        jsonObj = json.loads(line)
        dev.append([jsonObj['qID'],jsonObj['sentence'], jsonObj['option1'],jsonObj['option2'],jsonObj['answer']])

df_dev = pd.DataFrame(
  dev,
    columns=['ID','question', 'option1', 'option2','label']
) 
"""
with open('./test.jsonl') as dataFile:
    for line in dataFile.readlines():
        jsonObj = json.loads(line)
        test.append([jsonObj['qID'],jsonObj['sentence'], jsonObj['option1'],jsonObj['option2'],jsonObj['answer']])

df_test = pd.DataFrame(
  test,
    columns=['ID','question/context', 'option1', 'option22','label']
) 
"""
joined_dataset = df_train.append(df_dev)
#df_train.to_csv('./winogrande/train.csv', index=False)
#df_test.to_csv('./winogrande/test.csv', index=False)
#df_dev.to_csv('./winogrande/val.csv', index=False)

train, val, test = np.split(joined_dataset.sample(frac=1), [int(.6*len(joined_dataset)), int(.8*len(joined_dataset))])
train.to_csv('./winogrande/train.csv', index=False)
test.to_csv('./winogrande/test.csv', index=False)
val.to_csv('./winogrande/val.csv', index=False)
