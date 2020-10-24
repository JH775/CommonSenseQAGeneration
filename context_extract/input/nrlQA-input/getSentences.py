import csv
import pickle
import numpy as np
import pandas as pd

import os
import requests
import json
import re



### preparedata for NRL model #################################################

X = pd.read_csv('../../generated_context/context_500_ALL.csv')
print(X.context[0])
print(X.context)

with open('input_nrlQA.txt', 'w') as outfile:
    #json.dump(data, outfile)
    for row in X.context:
      
      x = {"sentence": row } 
      sentence = json.dumps(x)
      outfile.write(sentence+"\n")
outfile.close()


################################################################################
"""
X = pd.read_csv('../../generated_context/context_1000_ALL.csv')
print(X.context[0])
print(X.context)

with open('../input_QAG.txt', 'w') as outfile:
    #json.dump(data, outfile)
    for row in X.context:
        listx = row.split(".")
        for sentence in listx:
            if len(sentence.split(" ")) > 6:
                outfile.write(sentence+".\n")
outfile.close()
"""

### prepare single sentence data for NRL model

X = pd.read_csv('../../generated_context/context_500_ALL.csv')
print(X.context[0])
print(X.context)

with open('input_nrlQA_short.txt', 'w') as outfile:
    #json.dump(data, outfile)
    for row in X.context:
        listx = row.split(".")
        for sentence in listx:
            if len(sentence.split(" ")) > 6: 
                x = {"sentence": sentence.strip("\n") } 
                sentence = json.dumps(x)
                outfile.write(sentence+"\n")
outfile.close()





