#!/bin/python

import os

for a in [2]:
    for b in [0,1,2,3,4]:
        for m in ["ODB_POCLAT"]:
            os.system("python ./BNR.py %s %d %d"%(m,a,b))