import csv
import pickle
import numpy as np
import pandas as pd
import random
import os
import requests
import json
import re

# Combine final custom MC datasets 
final_nishanth = pd.read_csv('multiple_choice_nishanth.csv')
final_jah = pd.read_csv('multiple_choice_Jah.csv')
final_judith = pd.read_csv('multiple_choice_Judith.csv')

print(final_nishanth.columns)

#print(final_jah.columns)
print(final_judith.columns)
#
joined_dataset = final_nishanth.append(final_judith)
joined_dataset = joined_dataset.append(final_jah)
print(joined_dataset.shape)
joined_dataset['id'] = joined_dataset.index.astype(int)
joined_dataset['id'] = joined_dataset['id'].map(str) + '_custom'
answer_label = ['A','B','C','D','E']
whole_answer_label = ['A','B','C','D','E']
list_dataset = joined_dataset.values.tolist()
new_list = []
print (type(list_dataset[0][0]),type(list_dataset[0][1]))
for row in list_dataset:
    label = row[2]
    correct_answer = row[answer_label.index(label)]
    answer_label.remove(label)
    prev_info = [row[-1],str(row[0]) + str(row[1])]
    index = whole_answer_label.index(random.choice(answer_label))
    while row[3+index] == correct_answer:
        index = whole_answer_label.index(random.choice(answer_label))
    answers = [row[3+index],correct_answer]
    random.shuffle(answers)
    new_list.append(prev_info + answers+[answers.index(correct_answer)+1]) 
    answer_label.append(label)


df_new = pd.DataFrame(
  new_list,
    columns=['qID','sentence','option1', 'option2','answer']
) 
#df_new.reset_index(inplace=True)

#joined_dataset.to_csv('final_CS_dataset2.csv', index=False)
#pd.read_csv('data/QA_set2.csv')
print(df_new)


train, val, test = np.split(df_new.sample(frac=1), [int(.6*len(joined_dataset)), int(.8*len(joined_dataset))])
train.to_json('./multi_choice_custom/Winogrande_format/train.jsonl', orient='records')
val.to_json('./multi_choice_custom/Winogrande_format/val.jsonl', orient='records')
test.to_json('./multi_choice_custom/Winogrande_format/test.jsonl', orient='records')#
json_set = train.to_json( orient='records')
x = json.loads(json_set)[0]
print(x)

# multiple choice dataset in JSON format: 
"""
with open('./multi_choice_custom/Winogrande_format/train.jsonl','w') as dataFile:
    json_set = train.to_json( orient='records')
    data = json.loads(json_set)
    for j in data:
        dataFile.write(str(j))

with open('./multi_choice_custom/Winogrande_format/val.jsonl','w') as dataFile:
    json_set = val.to_json(orient='records')
    data = json.loads(json_set)
    for x in data:
        dataFile.write(str(j))
with open('./multi_choice_custom/Winogrande_format/test.jsonl','w') as dataFile:
    json_set = test.to_json( orient='records')
    data = json.loads(json_set)
    for x in data:
        dataFile.write(str(j))

"""
