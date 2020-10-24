import csv
import pickle
import numpy as np
import pandas as pd

import os
import requests



with open('input_cos_e.txt') as f:
    listY = f.read().split('\n')
    print(len(listY))

with open('input_noduplicates.txt') as f:
    listX = f.read().split('\n')
    print(len(listX))

with open('input_social_i_qa.txt') as f:
    listZ = f.read().split('\n')
    print(len(listZ))

list_all = listX + listY + listZ
print(list_all[6], list_all[14449], len(list_all))
df = pd.DataFrame(list_all,columns=['context'])
np.savetxt('input_ALL.txt', df.context,fmt='%s',delimiter='\t')
