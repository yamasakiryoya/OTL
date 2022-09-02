import os
import re
import numpy as np
import numpy.random as rd
import pandas as pd
from PIL import Image
rd.seed(0)
root_path = './'

i = 0
list_df = pd.DataFrame( columns=['file','age'] )
for picture_name in os.listdir(root_path + 'morph2-aligned/'):
    if picture_name.split('.')[-1] == 'jpg':
        if int(re.split('[MF]',picture_name)[-1].split('.')[0])-16<55:
            tmp_se = pd.Series( [picture_name, int(re.split('[MF]',picture_name)[-1].split('.')[0])-16], index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
            i+=1
            print(i)
list_df.to_csv('../morph2/morph2_all.csv')
