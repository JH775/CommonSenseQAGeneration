import csv
import pickle
import numpy as np
import pandas as pd

import os
import requests
import json
import re

final_nishanth = pd.read_csv('Final_data.csv')
final_jah = pd.read_csv('jmarkaba-final_data.csv')
# dataset generated from collected CS-dataset (https://github.com/sheng-z/) context and further adapted using BERT
final_judith = pd.read_csv('judith_final_1.csv')
# dataset generated from generated context
final_judith2 = pd.read_csv('judith_final2.csv')

final_judith2 = final_judith2.drop_duplicates()
labels = ['yes' for x in range(final_judith2.shape[0])]
final_judith2['labels'] = labels
print(len(labels))
print(final_nishanth.shape)
print(final_jah.shape)
print(final_judith.shape)
#
joined_dataset = final_nishanth.append(final_jah)
joined_dataset = joined_dataset.append(final_judith)
joined_dataset = joined_dataset.append(final_judith2)
print(joined_dataset.shape)
joined_dataset.to_csv('final_CS_dataset.csv', index=False)
#pd.read_csv('data/QA_set2.csv')