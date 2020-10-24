import csv
import pickle
import numpy as np
import pandas as pd

import os
import requests
import json
import re



# extract context, questions, answers, and start and end answer token from questions generated with nrl model
error_file = open('ERROR_data_2.txt','w')
df = pd.DataFrame(columns=['context','question','answer', 'start', 'end'])
context_valid = True

outfile = ['10-23.out']
with open('../generated_questions/'+ outfile[0], 'r') as infile:
    context= "start"
    while context:
        try:
            context = infile.readline().strip("input:").replace("{\'", "{\"").replace("\'}", "\"}").replace(", \'", ", \"").replace("\',", "\",").replace("\':", "\":").replace(": \'", ": \"")
            context = context.replace(" \" ", " \' ")
            context = json.loads(context)
            context_cell = context["sentence"]
        except ValueError: 
            context_valid = False
            error_file.write("ERROR"+ context)
        
        if(context_valid):
            try:
                prediction = infile.readline().strip("prediction:").replace("{\'", "{\"").replace("\'}", "\"}").replace(", \'", ", \"").replace("\',", "\",")
                prediction = prediction.replace(" \" ", " \' ")
                prediction = json.loads(prediction)

                verb_list = prediction['verbs']
                for verb in verb_list:
                    for question in verb['qa_pairs']:
                        question_cell = question['question']
                        answer_cell = question['spans'][0]['text'] 
                        answer_start = question['spans'][0]['start']
                        answer_end = question['spans'][0]['end']
                        df = df.append({'context': context_cell,'question' :question_cell ,'answer': answer_cell, 'start':answer_start , 'end': answer_end }, ignore_index=True)
            except ValueError:
                error_file.write("ERROR"+ prediction)
            finally:
                infile.readline()
                
        else:
            infile.readline()
            infile.readline()
            context_valid = True

df.to_csv('QA_set.csv', index=False)


with open('../generated_questions/out_nrlQA.txt', 'r') as infile:
    data= "start"
    while data:
        try:
            data = infile.readline()
            data = json.loads(data)
            context_cell = " ".join(data["words"])
            verb_list = data['verbs']
            if len(verb_list) != 0:
                for verb in verb_list:
                    for question in verb['qa_pairs']:
                        question_cell = question['question']
                        answer_cell = question['spans'][0]['text'] 
                        answer_start = question['spans'][0]['start']
                        answer_end = question['spans'][0]['end']
                        df = df.append({'context': context_cell,'question' : question_cell ,'answer': answer_cell, 'start':answer_start , 'end': answer_end }, ignore_index=True)
        except ValueError: 
            error_file.write("ERROR"+ data)
        

df.to_csv('QA_set_fixed.csv', index=False)










