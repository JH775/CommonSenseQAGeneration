from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath
import tempfile
import os
import pandas as pd
import gensim.downloader as api
from num2words import num2words
import random

# Set file names for train and test data
labels = ['A','B','C','D']
final_judith = pd.read_csv('QA_set2.csv')
answers = final_judith.drop_duplicates(subset=['answer'])['answer'].tolist()
#model = FastText()
model = api.load("glove-twitter-50") 
train = []
error = []
single_word_answer = []
def checkForNumbers(input):
    try:
        int(input)
        return num2words(input)
    except ValueError:
        try: 
            x = float(input)
            x = round(x,0)
            return num2words(str(x))
        except ValueError:
                return input
        return input


def findmostsimilar(word_array):
    max = 0
    array = []
    w_list = [checkForNumbers(w_i.lower()) for w_i in word_array.split(' ')]
    for a in answers:
        if a != word_array:
            try:
                a_list = [checkForNumbers(a_i.lower()) for a_i in a.split(' ')]
                
                #if(len(a_list) == 1):
                #    a_list = [a_list]
                x = model.n_similarity(a_list, w_list)
                if x > 0.8:
                    array.append((x,a))
            except KeyError:
                x = 1
    random.shuffle(array)
    return array[:3]
                


print(len(final_judith['answer'].tolist()))
print(type(final_judith['answer'].tolist()[0]))
# build the vocabulary

#model.build_vocab(final_nishanth['question'].tolist())
print('train')
#word_array = '1.67 times a week'
#w_list = [checkForNumbers(w_i) for w_i in word_array.split(' ')]
#print(w_list)
count = 0
no_count = 0
error_count = 0
single_word_count = 0
for i, row in final_judith.iterrows():
    answer = row['answer']
    question = row['question']
    context = row['context']
    if len(answer.split(" ")) == 1:
        try:
            wrong_answers = model.most_similar(answer)
        except KeyError:
            error.append([context,question,answer])
            error_count += 1
            continue
    else:         
        wrong_answers = findmostsimilar(answer)
    if len(wrong_answers) != 3:
        error.append([context,question,answer])
        error_count += 1
        continue
    else: 
        count += 1
  
    answer_set = [answer,wrong_answers[0][1],wrong_answers[1][1],wrong_answers[2][1]] 
    random.shuffle(answer_set)
    l = answer_set.index(answer)
    #if row['label'] == 'no':
        #error.append([context,question,answer,row['label']])
        #train.append([context,question,'E'] + answer_set + ['None of the above'])
        #no_count += 1
    #else:
    train.append([context,question,labels[l]] + answer_set + ['None of the above'])
    if count%100 == 0:
            print(count)

df_train = pd.DataFrame(
  train,
    columns=['context','question','label','answerA', 'answerB','answerC','answerD', 'answerE']
) 

df_error = pd.DataFrame(
  error,
    columns=['context','question','answer']
) 

df_single_word = pd.DataFrame(
  single_word_answer,
    columns=['context','question','answer']
) 
df_train.to_csv('multiple_choice_Judith.csv', index=False)
df_error.to_csv('remainder_Judith.csv', index=False)
print(count)
print(error_count)
print(no_count)





