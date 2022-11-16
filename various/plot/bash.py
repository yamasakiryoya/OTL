#import
import os

for a in [2]:
    for b in [0,1,2,3,4]:
        for m in ["AD"]:
            os.system("python ./NO.py %s %d %d"%(m,a,b))
        for m in ["ODOB_LogiIT"]:
            os.system("python ./SO.py %s %d %d"%(m,a,b))
        for m in ["ODOB_LogiAT","POOCL_NLL"]:
            os.system("python ./LSO.py %s %d %d"%(m,a,b))
        for m in ["ODB_LogiIT"]:
            os.system("python ./MSO.py %s %d %d"%(m,a,b))