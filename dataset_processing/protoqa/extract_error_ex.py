import csv
import pickle
import numpy as np
import pandas as pd
import gensim.downloader as api
import random

error = pd.read_csv('error.csv')

# extract more samples by trying to find more answers for each sample
model = api.load("word2vec-google-news-300") 

labels = ['A','B','C','D']
k = -1
print('start')
train = []
count = 0
for i, row in error.iterrows():
    answer = row['answer']
    question = row['question']
    print(question,answer)
    try:
        if '/' in answer and len(answer.split(' ')) < 2:
            false_answers = model.most_similar(answer.split('/')[0])
        else:
            false_answers = model.most_similar(answer)
        answer_set = [answer,false_answers[0][0],false_answers[1][0],false_answers[3][0]]
        random.shuffle(answer_set)
        l = answer_set.index(answer)
        train.append([question,labels[l]] + answer_set)
        k += 1
    except KeyError:
        print(question,answer)
        count += 1

    if i%10 == 0:
        print(i)
        
print("total error count: ",count)
print("predicted count ", k  )
