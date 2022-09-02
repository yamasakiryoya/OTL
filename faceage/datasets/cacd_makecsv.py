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
list_df.to_csv('../cacd/cacd_all.csv')
