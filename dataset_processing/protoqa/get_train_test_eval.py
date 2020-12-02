import csv
import pickle
import numpy as np
import pandas as pd

# split dataset into train,val, and error
train = pd.read_csv('train.csv')
dev = pd.read_csv('dev.csv')
joined_dataset = train.append(dev)

print(joined_dataset.shape)
joined_dataset['id'] = joined_dataset.index.astype(int)
joined_dataset['id'] = joined_dataset['id'].map(str) + '_protoqa'

train, val, test = np.split(joined_dataset.sample(frac=1), [int(.6*len(joined_dataset)), int(.8*len(joined_dataset))])
train.to_csv('./final/protoqa/train.csv', index=False)
val.to_csv('./final/protoqa/val.csv', index=False)
test.to_csv('./final/protoqa/test.csv', index=False)#


