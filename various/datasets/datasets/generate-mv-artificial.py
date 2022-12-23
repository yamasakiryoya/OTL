import numpy as np
import numpy.random as rd
rd.seed(0)

N = 40768
X = np.zeros((N,11))

X[:,0] = rd.uniform(-5,5,N)
X[:,1] = rd.uniform(-15,-10,N)
for i in range(N):
    if X[i,0]>0:
        X[i,2]=0#green
        X[i,3]=(X[i,0]+2.*X[i,1])
    else:
        if rd.rand()<0.4:
            X[i,2]=1#red
        else:
            X[i,2]=2#brown (I think that X4=brown in https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html is mis. I operate X3=brown.)
        if rd.rand()<0.3:
            X[i,3]=X[i,0]/2.
        else:
            X[i,3]=X[i,1]/2.
X[:,4] = rd.uniform(-1,1,N)
X[:,5] = X[:,3]*rd.uniform(0,5,N)
for i in range(N):
    if rd.rand()<0.3:
        X[i,6]=0#yes
    else:
        X[i,6]=1#no
for i in range(N):
    if X[i,4]<0.5:
        X[i,7]=0#normal
    else:
        X[i,7]=1#large
X[:,8] = rd.uniform(100,500,N)
X[:,9] = rd.uniform(1000,1200,N)

for i in range(N):
    if X[i,1]>2:
        X[i,10]=35-0.5*X[i,3]
    elif -2<=X[i,3]<=2:
        X[i,10]=10-2*X[i,0]
    elif X[i,6]==0:
        X[i,10]=3-X[i,0]/X[i,3]
    elif X[i,7]==0:
        X[i,10]=X[i,5]+X[i,0]
    else:
        X[i,10]=X[i,0]/2

np.savetxt("./mv-artificial.csv", X, delimiter=",")