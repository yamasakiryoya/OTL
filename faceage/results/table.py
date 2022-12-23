import os
import numpy as np

T  = ['Z','A','S']
D  = ['morph2','cacd','afad']
M  = ['AD','hing-IT','ord-hing-IT','ord-hing-AT','logi-IT','ord-logi-IT','ord-logi-AT','POOCL-NLL']
L  = [['NT','OT'],['MT','ST','OT'],['ST','OT'],['ST','OT'],['MT','ST','OT'],['ST','OT'],['ST','RL','OT'],['ST','RL','OT']]
NUM = 10

RES, MEA, STD = np.zeros((3,3,20,NUM)), np.zeros((3,3,20)), np.zeros((3,3,20))

for t in range(len(T)):
    for d in range(len(D)):
        i = 0
        for m in range(len(M)):
            for l in L[m]:
                for seed in range(NUM):
                    tmp = np.loadtxt("../%s/result/%s/seed%d/training_%s.log"%(D[d],M[m],seed,l), delimiter=",")
                    if t==2: tmp = np.sqrt(tmp)
                    RES[t,d,i,seed] = tmp[np.argmin(tmp[:,3+t]),6+t]
                i += 1


for t in range(len(T)):
    for d in range(len(D)):
        for i in range(20):
            MEA[t,d,i] = np.mean(RES[t,d,i,:])
            STD[t,d,i] = np.std(RES[t,d,i,:])

for t in range(len(T)):
    for i in range(20):
        for d in range(len(D)):
            print("$%.4f_{%.4f}$ & "%(MEA[t,d,i],STD[t,d,i]),end='')
        print("")
    print("")


for t in range(len(T)):
    for d in range(len(D)):
        print("%d,%d"%(t,d),end=" ")
        for i in range(20):
            if np.min(MEA[t,d,:])==MEA[t,d,i]:
                print(i, end=" ")
        print("")
    print("")

from scipy import stats

for t in range(len(T)):
    for d in range(len(D)):
        print(t,d,np.argmin(np.array([MEA[t,d,0],MEA[t,d,1]])),end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,0,:], RES[t,d,1,:], alternative='greater').pvalue)

    for d in range(len(D)):
        print(t,d,np.argmin(np.array([MEA[t,d,2],MEA[t,d,3],MEA[t,d,4]])),end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,2,:], RES[t,d,3,:], alternative='greater').pvalue,end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,2,:], RES[t,d,4,:], alternative='greater').pvalue,end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,3,:], RES[t,d,4,:], alternative='greater').pvalue)

    for d in range(len(D)):
        print(t,d,np.argmin(np.array([MEA[t,d,5],MEA[t,d,6]])),end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,5,:], RES[t,d,6,:], alternative='greater').pvalue)

    for d in range(len(D)):
        print(t,d,np.argmin(np.array([MEA[t,d,7],MEA[t,d,8]])),end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,7,:], RES[t,d,8,:], alternative='greater').pvalue)

    for d in range(len(D)):
        print(t,d,np.argmin(np.array([MEA[t,d,9],MEA[t,d,10],MEA[t,d,11]])),end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,9,:], RES[t,d,10,:], alternative='greater').pvalue,end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,9,:], RES[t,d,11,:], alternative='greater').pvalue,end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,10,:], RES[t,d,11,:], alternative='greater').pvalue)

    for d in range(len(D)):
        print(t,d,np.argmin(np.array([MEA[t,d,12],MEA[t,d,13]])),end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,12,:], RES[t,d,13,:], alternative='greater').pvalue)

    for d in range(len(D)):
        print(t,d,np.argmin(np.array([MEA[t,d,14],MEA[t,d,15],MEA[t,d,16]])),end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,14,:], RES[t,d,15,:], alternative='greater').pvalue,end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,14,:], RES[t,d,16,:], alternative='greater').pvalue,end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,15,:], RES[t,d,16,:], alternative='greater').pvalue)

    for d in range(len(D)):
        print(t,d,np.argmin(np.array([MEA[t,d,17],MEA[t,d,18],MEA[t,d,19]])),end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,17,:], RES[t,d,18,:], alternative='greater').pvalue,end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,17,:], RES[t,d,19,:], alternative='greater').pvalue,end="  ")
        print("%.4f"%stats.mannwhitneyu(RES[t,d,18,:], RES[t,d,19,:], alternative='greater').pvalue)