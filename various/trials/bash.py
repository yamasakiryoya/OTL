#!/bin/python

import os

for a in [2]:
    for b in [4]:
        #"AD"
        for m in ["AD"]:
            os.system("python ./NNR.py %s %d %d"%(m,a,b))
        #"ODB_POCLAT","ODOB_POCLAT","ODB_SVORIT","ODOB_SVORIT"
        for m in ["ODB_POCLAT","ODOB_POCLAT","ODB_SVORIT","ODOB_SVORIT"]:
            os.system("python ./BNR.py %s %d %d"%(m,a,b))
        #"POCL_NLL","POOCL_NLL"
        for m in ["POCL_NLL","POOCL_NLL"]:
            os.system("python ./BSR.py %s %d %d"%(m,a,b))