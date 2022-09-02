import os
import numpy as np
import numpy.random as rd
import pandas as pd
from PIL import Image
rd.seed(0)
root_path = './tarball/AFAD-Full/'

i = 0
list_df = pd.DataFrame(columns=['path','age'])
for age in range(15,41):
    for mw in [111,112]:
        path = root_path + str(age) + '/' + str(mw)
        path2 = str(age) + '/' + str(mw)
        for picture_name in os.listdir(path):
            if picture_name.split('.')[-1] == 'jpg':
                tmp_se = pd.Series( [path2 + '/' + picture_name, int(age-15)], index=list_df.columns )
                list_df = list_df.append( tmp_se, ignore_index=True )
                i+=1
                print(i)
list_df.to_csv('../afad/afad_all.csv')
