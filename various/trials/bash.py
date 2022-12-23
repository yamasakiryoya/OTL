#!/bin/python
import os

for a in range(7):
    for b in range(29):
        if a>0 or b<17:
            os.system("python ./NO.py AD %d %d"%(a,b))
            os.system("python ./SO.py ODOB_LogiIT %d %d"%(a,b))
            os.system("python ./SO.py ODOB_HingIT %d %d"%(a,b))
            os.system("python ./SO.py ODOB_HingAT %d %d"%(a,b))
            os.system("python ./LSO.py ODOB_LogiAT %d %d"%(a,b))
            os.system("python ./LSO.py POOCL_NLL %d %d"%(a,b))
            os.system("python ./MSO.py ODB_LogiIT %d %d"%(a,b))
            os.system("python ./MSO.py ODB_HingIT %d %d"%(a,b))
