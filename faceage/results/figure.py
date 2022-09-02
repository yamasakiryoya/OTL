import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 15
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'k'

color = ['r','g','b','c','gold','m', 'k', 'gray']
D = ['afad', 'cacd', 'morph2']
D2 = ['AFAD', 'CACD', 'MORPH-2']
T = ['A', 'S']#['Z', 'A', 'S']

#===== Non-ordered =====#
for d in range(len(D)):
    data = D[d]
    data2 = D2[d]
    for t in range(len(T)):
        task = T[t]
        #
        fig=plt.figure(figsize=(12,5.5))
        ax = fig.add_subplot(111)
        tmp = np.loadtxt("../%s/MTM/LC/training.log"%data, delimiter=",")
        for i in range(200):
            if tmp[i,-(3-t)]==0:
                ax.axvspan(i+0.5, i+1.5, color="gray", edgecolor=None, alpha=0.5)
        plt.plot(range(1,201), tmp[:, 3+t], linewidth=2, c=color[0], linestyle="-", label="MT, ST, training")
        plt.plot(range(1,201), tmp[:, 0+t], linewidth=2, c=color[1], linestyle="--", label="LB, training")
        plt.plot(range(1,201), tmp[:, 6+t], linewidth=2, c=color[2], linestyle="-", label="IOT, training")
        plt.plot(range(1,201), tmp[:,21+t], linewidth=2, c=color[3], linestyle="-", label="MT, ST, test")
        plt.plot(range(1,201), tmp[:,18+t], linewidth=2, c=color[4], linestyle="--", label="LB, test")
        plt.plot(range(1,201), tmp[:,24+t], linewidth=2, c=color[5], linestyle="-", label="IOT, test")
        if t==0: plt.text(0.5, 0.95, '%s, MZE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
        if t==1: plt.text(0.5, 0.95, '%s, MAE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
        if t==2: plt.text(0.5, 0.95, '%s, RMSE'%data2,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
        plt.grid()
        plt.xlim(-1,202)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("./%s-%s-%s.png"%(data,task),bbox_inches="tight", pad_inches=0.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()


