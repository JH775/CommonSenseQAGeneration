from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath
import tempfile
import os
import pandas as pd
import gensim.downloader as api
from num2words import num2words
import random

labels = ['A','B','C','D','E']
winogrande = pd.read_csv('remainder_val_Judith.csv')
#model = FastText()
model = api.load("glove-twitter-25") 
answers = winogrande.drop_duplicates(subset=['option2'])['option2'].tolist()
train = []
error = []
single_word_answer = []
def checkForNumbers(input):
    """ convert integers and floats into words"""
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
    """ Find most similar word vectors from list of all word vectors""" 
    max = 0
    array = []
    w_list = [checkForNumbers(w_i.lower()) for w_i in word_array.split(' ')]
    for a in answers:
        if a != word_array:
            try:
                a_list = [checkForNumbers(a_i.lower()) for a_i in a.split(' ')]
                
                x = model.n_similarity(a_list, w_list)
                if x > 0.8:
                    array.append((x,a))
            except KeyError:
                x = 1
    random.shuffle(array)
    return array[:3]
                


print('Get answers: ')

count = 0
no_count = 0
error_count = 0
single_word_count = 0
for i, row in winogrande.iterrows():
    label = row['label']
    id_x = row['ID']
    if label == 1:
        right_answer = row['option1']
        wrong_answer = row['option2']
    else:
        right_answer = row['option2']
        wrong_answer = row['option1']
    question = row['question']
    if len(right_answer.split(" ")) == 1:
        try:
            wrong_answers = model.most_similar(right_answer)
        except KeyError:
           wrong_answers = findmostsimilar(right_answer)
    else:
        wrong_answers = findmostsimilar(right_answer)
    if len(wrong_answers) != 3:
        error.append([id_x,question,label,right_answer,wrong_answer])
        error_count += 1
        continue
    else: 
        count += 1
  
    answer_set = [right_answer,wrong_answer, wrong_answers[0][1],wrong_answers[1][1],wrong_answers[2][1]] 
    random.shuffle(answer_set)
    l = answer_set.index(right_answer)
    train.append([id_x,question,labels[l]] + answer_set)
    if count%100 == 0:
            print(count)

df_train = pd.DataFrame(
  train,
    columns=['ID','question','label','answerA', 'answerB','answerC','answerD', 'answerE']
) 

df_error = pd.DataFrame(
  error,
    columns=['ID','question','label','option1','option2']
) 
df_train.to_csv('val_MC2.csv', index=False)
df_error.to_csv('remainder_val.csv', index=False)
print(count)
print(error_count)
print(no_count)