import os
import numpy as np
import numpy.random as rd
import pandas as pd
from PIL import Image
rd.seed(0)
root_path = './'

i = 0
list_df = pd.DataFrame( columns=['file','age'] )
for picture_name in os.listdir(root_path + 'CACD2000-centered/'):
    if picture_name.split('.')[-1] == 'jpg':
        tmp_se = pd.Series( [picture_name, int(picture_name.split('_')[0])-14], index=list_df.columns )
        list_df = list_df.append( tmp_se, ignore_index=True )
        i+=1
        print(i)
list_df.to_csv('./cacd_all.csv')

NUM_ALL = i
NUM_TRA = int(NUM_ALL*0.72)
NUM_VAL = int(NUM_ALL*0.08)
NUM_TES = NUM_ALL - NUM_TRA - NUM_VAL

with open('./cacd_all.csv') as f:
    l = f.readlines()
    l = l[1:]
    TRA = rd.choice(range(NUM_ALL), NUM_TRA, replace = False)
    rest = list(set(range(NUM_ALL)) - set(TRA))
    VAL = rd.choice(rest, NUM_VAL, replace = False)
    TES = rd.permutation(np.array(list(set(rest) - set(VAL))))
    l_TRA, l_VAL, l_TES = [',file,age\n'], [',file,age\n'], [',file,age\n']
    for i in TRA: l_TRA.append(l[i])
    for i in VAL: l_VAL.append(l[i])
    for i in TES: l_TES.append(l[i])
    with open('../cacd/cacd_train.csv', mode='w') as g: g.writelines(l_TRA)
    with open('../cacd/cacd_valid.csv', mode='w') as g: g.writelines(l_VAL)
    with open('../cacd/cacd_test.csv', mode='w') as g: g.writelines(l_TES)