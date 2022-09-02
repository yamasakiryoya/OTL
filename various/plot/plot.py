import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'k'

color = ['r','g','b','c','gold','m','k','gray']
T = ['Z', 'A', 'S']
D  = ["LEV","ERA","SWD","winequality-red","car"]
D2 = ["LEV","ERA","SWD","WQR","CAR"]
M  = ['ODB_POCLAT']
M2 = ['OLR-ATa']

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
            if len(tmp[:,0])==200:
                print(data,task,method)
                fig=plt.figure(figsize=(18,5.5))
                ax = fig.add_subplot(111)
                for i in range(200):
                    if tmp[i,-(3-t)]==0:
                        ax.axvspan(i+0.5, i+1.5, color="gray", edgecolor=None, alpha=0.5)
                if t<2:
                    plt.plot(range(1,201), tmp[:, 3+t], linewidth=3, c=color[0], label="MT,ST, training")
                    plt.plot(range(1,201), tmp[:, 0+t], linewidth=3, c=color[1], linestyle="--", label="LB, training")
                    plt.plot(range(1,201), tmp[:, 9+t], linewidth=3, c=color[2], label="IOT, training")
                    plt.plot(range(1,201), tmp[:,27+t], linewidth=3, c=color[3], label="MT,ST, test")
                    plt.plot(range(1,201), tmp[:,24+t], linewidth=3, c=color[4], linestyle="--", label="LB, test")
                    plt.plot(range(1,201), tmp[:,33+t], linewidth=3, c=color[5], label="IOT, test")
                else:
                    plt.plot(range(1,201), np.sqrt(tmp[:, 3+t]), linewidth=3, c=color[0], label="MT,ST, training")
                    plt.plot(range(1,201), np.sqrt(tmp[:, 0+t]), linewidth=3, c=color[1], linestyle="--", label="LB, training")
                    plt.plot(range(1,201), np.sqrt(tmp[:, 9+t]), linewidth=3, c=color[2], label="IOT, training")
                    plt.plot(range(1,201), np.sqrt(tmp[:,27+t]), linewidth=3, c=color[3], label="MT,ST, test")
                    plt.plot(range(1,201), np.sqrt(tmp[:,24+t]), linewidth=3, c=color[4], linestyle="--", label="LB, test")
                    plt.plot(range(1,201), np.sqrt(tmp[:,33+t]), linewidth=3, c=color[5], label="IOT, test")
                if t==0: plt.text(0.5, 0.95, '%s, MZE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=30)
                if t==1: plt.text(0.5, 0.95, '%s, MAE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=30)
                if t==2: plt.text(0.5, 0.95, '%s, RMSE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=30)
                plt.grid()
                plt.xlim(-1,202)
                plt.tight_layout()
                if os.path.isdir("./plot")==False: os.makedirs("./plot")
                plt.savefig("./plot/%s-%s-%s.png"%(data,task,method2),bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close()
