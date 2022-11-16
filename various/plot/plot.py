#import
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'k'

cmap = plt.get_cmap('jet')
color = ["r","g","b","c","m"]
T = ['Z', 'A', 'S']
D  = ["LEV","ERA","SWD","winequality-red","car"]
D2 = ["LEV","ERA","SWD","WQR","CAR"]
M  = ["AD","ODOB_LogiIT","ODOB_LogiAT","POOCL_NLL","ODB_LogiIT"]
M2 = ["AD","Logistic-IT,ordered","Logistic-AT,ordered","OLR-NLL","Logistic-IT,non-ordered"]

for d in range(len(D)):
    data = D[d]
    data2 = D2[d]
    for t in range(len(T)):
        task = T[t]
        for m in range(len(M)):
            method = M[m]
            method2 = M2[m]
            #
            tmp = np.loadtxt("./Results/%s-%s.csv"%(method,data), delimiter=",")
            if len(tmp[:,0])==500:
                print(data,task,method)
                fig=plt.figure(figsize=(15,6))
                ax = fig.add_subplot(111)
                if t==2:
                    tmp=np.sqrt(tmp)
                if m==0:
                    plt.plot(range(1,501), tmp[:, 0+t], linewidth=2, c=color[0], label="NNT, training")
                    plt.plot(range(1,501), tmp[:, 3+t], linewidth=2, c=color[4], label="EOT, training")
                    plt.plot(range(1,501), tmp[:,12+t], linewidth=2, c=color[0], linestyle="--", label="NNT, test")
                    plt.plot(range(1,501), tmp[:,15+t], linewidth=2, c=color[4], linestyle="--", label="EOT, test")
                    Mi = np.min([np.min(tmp[5:, 0+t]),np.min(tmp[5:, 3+t]),np.min(tmp[5:,12+t]),np.min(tmp[5:,15+t])])
                    Ma = np.max([np.max(tmp[5:, 0+t]),np.max(tmp[5:, 3+t]),np.max(tmp[5:,12+t]),np.max(tmp[5:,15+t])])
                elif m==1:
                    plt.plot(range(1,501), tmp[:, 0+t], linewidth=2, c=color[1], label="MT, ST, training")
                    plt.plot(range(1,501), tmp[:, 3+t], linewidth=2, c=color[4], label="EOT, training")
                    plt.plot(range(1,501), tmp[:,12+t], linewidth=2, c=color[1], linestyle="--", label="MT, ST, test")
                    plt.plot(range(1,501), tmp[:,15+t], linewidth=2, c=color[4], linestyle="--", label="EOT, test")
                    Mi = np.min([np.min(tmp[5:, 0+t]),np.min(tmp[5:, 3+t]),np.min(tmp[5:,12+t]),np.min(tmp[5:,15+t])])
                    Ma = np.max([np.max(tmp[5:, 0+t]),np.max(tmp[5:, 3+t]),np.max(tmp[5:,12+t]),np.max(tmp[5:,15+t])])
                elif m==2 or m==3:
                    plt.plot(range(1,501), tmp[:, 0+t], linewidth=2, c=color[1], label="MT, ST, training")
                    plt.plot(range(1,501), tmp[:, 3+t], linewidth=2, c=color[3], label="LB, training")
                    plt.plot(range(1,501), tmp[:, 6+t], linewidth=2, c=color[4], label="EOT, training")
                    plt.plot(range(1,501), tmp[:,18+t], linewidth=2, c=color[1], linestyle="--", label="MT, ST, test")
                    plt.plot(range(1,501), tmp[:,21+t], linewidth=2, c=color[3], linestyle="--", label="LB, test")
                    plt.plot(range(1,501), tmp[:,24+t], linewidth=2, c=color[4], linestyle="--", label="EOT, test")
                    Mi = np.min([np.min(tmp[5:, 0+t]),np.min(tmp[5:, 3+t]),np.min(tmp[5:, 6+t]),np.min(tmp[5:,18+t]),np.min(tmp[5:,21+t]),np.min(tmp[5:,24+t])])
                    Ma = np.max([np.max(tmp[5:, 0+t]),np.max(tmp[5:, 3+t]),np.max(tmp[5:, 6+t]),np.max(tmp[5:,18+t]),np.max(tmp[5:,21+t]),np.max(tmp[5:,24+t])])
                else:
                    plt.plot(range(1,501), tmp[:, 0+t], linewidth=2, c=color[1], label="MT, training")
                    plt.plot(range(1,501), tmp[:, 3+t], linewidth=2, c=color[2], label="ST, training")
                    plt.plot(range(1,501), tmp[:, 6+t], linewidth=2, c=color[4], label="EOT, training")
                    plt.plot(range(1,501), tmp[:,18+t], linewidth=2, c=color[1], linestyle="--", label="MT, test")
                    plt.plot(range(1,501), tmp[:,21+t], linewidth=2, c=color[2], linestyle="--", label="ST, test")
                    plt.plot(range(1,501), tmp[:,24+t], linewidth=2, c=color[4], linestyle="--", label="EOT, test")
                    Mi = np.min([np.min(tmp[5:, 0+t]),np.min(tmp[5:, 3+t]),np.min(tmp[5:, 6+t]),np.min(tmp[5:,18+t]),np.min(tmp[5:,21+t]),np.min(tmp[5:,24+t])])
                    Ma = np.max([np.max(tmp[5:, 0+t]),np.max(tmp[5:, 3+t]),np.max(tmp[5:, 6+t]),np.max(tmp[5:,18+t]),np.max(tmp[5:,21+t]),np.max(tmp[5:,24+t])])
                plt.grid()
                plt.ylim(np.max([0,Mi-0.1*(Ma-Mi)]),Ma+0.1*(Ma-Mi))
                plt.xlim(-1,502)
                plt.tight_layout()
                if os.path.isdir("./Results/plot")==False: os.makedirs("./Results/plot")
                plt.savefig("./Results/plot/%s-%s-%s.png"%(data,task,method2),bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close()
