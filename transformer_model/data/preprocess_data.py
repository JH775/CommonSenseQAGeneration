import pandas as pd
import numpy as np


final = pd.read_csv('./judith_final_1.csv')
final['index1'] = final.index

train, val, test = np.split(final.sample(frac=1), [int(.6*len(final)), int(.8*len(final))])
train.to_csv('./data/train.csv', index=False)
val.to_csv('./data/val.csv', index=False)
test.to_csv('./data/test.csv', index=False)
