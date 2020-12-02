import pandas as pd
import json
from num2words import num2words
import gensim.downloader as api
import random
train = []
dev = []
answer_sets = []
info = api.info() 
model = api.load("glove-twitter-25") 

# generate more answers using gensim API functions 

def checkForNumbers(input):
    if input.isnumeric(): 
        return num2words(input) 
    return input
useless = []
train = []
labels = ['A','B','C','D']
#curl --url https://raw.githubusercontent.com/iesl/protoqa-data/master/data/train/protoqa_train.jsonl -O 
#curl --url https://raw.githubusercontent.com/iesl/protoqa-data/master/data/dev/protoqa_scraped_dev.jsonl -O  

with open('protoqa_train.jsonl') as dataFile:
    i = -1
    for line in dataFile.readlines():
        jsonObj = json.loads(line)
        question = jsonObj['question']['normalized-question']

        answer_set = [checkForNumbers(x['answers'][0]) for x in jsonObj['answer-clusters']]
        answer = answer_set.pop(0)
        first = True
        quitit = False
        while len(answer_set) != 0 or quitit:
            try:
                false_answers = model.most_similar(answer)
                answer_set = [answer,false_answers[0][0],false_answers[1][0],false_answers[3][0]]
                random.shuffle(answer_set)
                l = answer_set.index(answer)
                train.append([question,labels[l]] + answer_set)
                i += 1
                quitit = True
                break
            except KeyError:
                if first and not quitit:
                    useless.append([question,answer, len(answer.split(' '))])
                    first = False
                answer = answer_set.pop(0)
            
        if i%1000 == 0:
            print(i)
        
        #answer_sets += answer_set 



df_train = pd.DataFrame(
  train,
    columns=['question','label','answerA', 'answerB','answerC','answerD']
) 

with open('protoqa_scraped_dev.jsonl') as dataFile:
    i = -1
    for line  in dataFile.readlines():
        jsonObj = json.loads(line)
        question = jsonObj['question']['normalized-question']
        answer_set = [checkForNumbers(x['answers'][0]) for x in jsonObj['answer-clusters']]
        answer = answer_set.pop(0)
        first = True
        while len(answer_set) != 0:
            try:
                false_answers = model.most_similar(answer)
                answer_set = [answer,false_answers[0][0],false_answers[1][0],false_answers[3][0]]
                random.shuffle(answer_set)
                l = answer_set.index(answer)
                dev.append([question,labels[l]] + answer_set)
                i += 1
                break
            except KeyError:
                if first:
                    useless.append([question,answer, len(answer.split(' '))])
                    first = False
                answer = answer_set.pop(0)
               
        if i%1000 == 0:
            print(i)
        
        answer_sets += answer_set 



df_dev = pd.DataFrame(
  dev,
    columns=['question','label','answerA', 'answerB','answerC','answerD']
) 


df_dev.to_csv('./dev.csv', index=False,encoding='utf-8')
df_train.to_csv('./train.csv', index=False,encoding='utf-8')

"""
with open('./ERROR_dev.txt', 'w') as file_handler:
    for item in useless:
        pair =  item[0] + '---' + item[1] + '--->'+ str(item[2])
        file_handler.write('{}\n'.format(pair))
"""
df_error = pd.DataFrame(
  useless,
    columns=['question','answer','length']
) 
df_error.to_csv('./error.csv', index=False,encoding='utf-8')
#print(df_train)
