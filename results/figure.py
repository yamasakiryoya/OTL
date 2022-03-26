import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'k'

color = ['r','g','b','c','gold','m','k','gray']
T = ['Z', 'A', 'S']
D  = ['afad', 'cacd', 'morph2']
D2 = ['AFAD', 'CACD', 'MORPH-2']
M  = ['ANLCL']
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
            tmp = np.loadtxt("../%s/MTM/%s-LC/training.log"%(data,method), delimiter=",")
            if len(tmp[:,0])==200:
                print(data,task,method)
                fig=plt.figure(figsize=(18,6))
                ax = fig.add_subplot(111)
                for i in range(200):
                    if tmp[i,-(3-t)]==0:
                        ax.axvspan(i+0.5, i+1.5, color="gray", edgecolor=None, alpha=0.5)
                plt.plot(range(1,201), tmp[:, 6+t], linewidth=3, c=color[0], label="MT,ST, training")
                plt.plot(range(1,201), tmp[:, 3+t], linewidth=3, c=color[1], linestyle="--", label="LB, training")
                plt.plot(range(1,201), tmp[:,12+t], linewidth=3, c=color[2], label="IOT, training")
                plt.plot(range(1,201), tmp[:,30+t], linewidth=3, c=color[3], label="MT,ST, test")
                plt.plot(range(1,201), tmp[:,27+t], linewidth=3, c=color[4], linestyle="--", label="LB, test")
                plt.plot(range(1,201), tmp[:,36+t], linewidth=3, c=color[5], label="IOT, test")
                if t==0: plt.text(0.5, 0.95, '%s, MZE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=30)
                if t==1: plt.text(0.5, 0.95, '%s, MAE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=30)
                if t==2: plt.text(0.5, 0.95, '%s, RMSE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=30)
                plt.grid()
                plt.xlim(-1,202)
                plt.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig("./MTM/Tsk/%s-%s-%s.png"%(data,task,method2),bbox_inches="tight", pad_inches=0.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.savefig("./MTM/Tsk/leg.png",bbox_inches="tight", pad_inches=0.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close()
