import os
import numpy as np


files = os.listdir("./datasets/")

for s in [3,5,10]:
    for f in files:
        print(s,f)
        if f.split(".")[-1]=="csv":
            data = np.loadtxt("./datasets/"+f, delimiter = ",")
            for j in range(len(data[0,:-1])):
                if np.std(data[:,j])!=0:
                    data[:,j] = (data[:,j]-np.mean(data[:,j]))/np.std(data[:,j])
                else:
                    data[:,j] = 0
            ind = np.argsort(data[:,-1])
            for k in range(s):
                for i in ind[int(k*(len(data[:,-1])//s)):np.max([int((k+1)*(len(data[:,-1])//s)),len(data[:,-1])])]:
                    data[i,-1] = k
            if os.path.exists("./EF%s/"%s)==False:
                os.makedirs("./EF%s/"%s)
            np.savetxt("./EF%s/"%s+"EF%s-"%s+f, data, delimiter=",")

for s in [3,5,10]:
    for f in files:
        print(s,f)
        if f.split(".")[-1]=="csv":
            data = np.loadtxt("./datasets/"+f, delimiter = ",")
            for j in range(len(data[0,:-1])):
                if np.std(data[:,j])!=0:
                    data[:,j] = (data[:,j]-np.mean(data[:,j]))/np.std(data[:,j])
                else:
                    data[:,j] = 0
            Mi = np.min(data[:,-1])
            Ma = np.max(data[:,-1])
            tmp = np.zeros(len(data[:,-1]))
            for i in range(len(data[:,-1])):
                for k in range(s):
                    if Mi+(Ma-Mi)/s*k<=data[i,-1]<=Mi+(Ma-Mi)/s*(k+1):
                        tmp[i] = k
            data[:,-1] = tmp
            if os.path.exists("./EL%s/"%s)==False:
                os.makedirs("./EL%s/"%s)
            np.savetxt("./EL%s/"%s+"EL%s-"%s+f, data, delimiter=",")