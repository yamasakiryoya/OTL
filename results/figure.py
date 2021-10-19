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
M = ['SVOR', 'NLL', 'ANLCL']
M2 = ['SVOR', 'OLR-NLL', 'OLR-ANLCL']


#===== Non-ordered =====#
for d in range(len(D)):
    data = D[d]
    data2 = D2[d]
    for t in range(len(T)):
        task = T[t]
        for m in range(len(M)):
            method = M[m]
            method2 = M2[m]
            fig=plt.figure(figsize=(12,6))
            ax = fig.add_subplot(111)
            tmp = np.loadtxt("../%s/threshold/%s-LC/training.log"%(data,method), delimiter=",")
            for i in range(100):
                if tmp[i,-(3-t)]==0:
                    ax.axvspan(i+0.5, i+1.5, color="gray", edgecolor=None, alpha=0.5)
            if m==0:
                plt.plot(range(1,101), tmp[:, 0+t], linewidth=2, c=color[0], linestyle="-", label="CT, training")
                plt.plot(range(1,101), tmp[:, 3+t], linewidth=2, c=color[2], linestyle="-", label="ROT, training")
                plt.plot(range(1,101), tmp[:,12+t], linewidth=2, c=color[3], linestyle="-", label="CT, test")
                plt.plot(range(1,101), tmp[:,15+t], linewidth=2, c=color[5], linestyle="-", label="ROT, test")
            else:
                plt.plot(range(1,101), tmp[:, 3+t], linewidth=2, c=color[0], linestyle="-", label="CT, training")
                plt.plot(range(1,101), tmp[:, 0+t], linewidth=2, c=color[1], linestyle="--", label="SMB, training")
                plt.plot(range(1,101), tmp[:, 6+t], linewidth=2, c=color[2], linestyle="-", label="ROT, training")
                plt.plot(range(1,101), tmp[:,21+t], linewidth=2, c=color[3], linestyle="-", label="CT, test")
                plt.plot(range(1,101), tmp[:,18+t], linewidth=2, c=color[4], linestyle="--", label="SMB, test")
                plt.plot(range(1,101), tmp[:,24+t], linewidth=2, c=color[5], linestyle="-", label="ROT, test")
            if t==0: plt.text(0.5, 0.95, '%s, Off, %s, MZE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
            if t==1: plt.text(0.5, 0.95, '%s, Off, %s, MAE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
            if t==2: plt.text(0.5, 0.95, '%s, Off, %s, RMSE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
            plt.grid()
            plt.xlim(-1,102)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig("./off/%s-%s-%s.png"%(data,task,method),bbox_inches="tight", pad_inches=0.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()


#===== Ordered =====#
for d in range(len(D)):
    data = D[d]
    data2 = D2[d]
    for t in range(len(T)):
        task = T[t]
        for m in range(len(M)):
            method = M[m]
            method2 = M2[m]
            fig=plt.figure(figsize=(12,6))
            ax = fig.add_subplot(111)
            tmp = np.loadtxt("../%s/threshold/ordered-%s-LC/training.log"%(data,method), delimiter=",")
            for i in range(100):
                if tmp[i,-(3-t)]==0:
                    ax.axvspan(i+0.5, i+1.5, color="gray", edgecolor=None, alpha=0.5)
            if m==0:
                plt.plot(range(1,101), tmp[:, 0+t], linewidth=2, c=color[0], linestyle="-", label="CT, training")
                plt.plot(range(1,101), tmp[:, 3+t], linewidth=2, c=color[2], linestyle="-", label="ROT, training")
                plt.plot(range(1,101), tmp[:,12+t], linewidth=2, c=color[3], linestyle="-", label="CT, test")
                plt.plot(range(1,101), tmp[:,15+t], linewidth=2, c=color[5], linestyle="-", label="ROT, test")
            else:
                plt.plot(range(1,101), tmp[:, 3+t], linewidth=2, c=color[0], linestyle="-", label="CT, training")
                plt.plot(range(1,101), tmp[:, 0+t], linewidth=2, c=color[1], linestyle="--", label="SMB, training")
                plt.plot(range(1,101), tmp[:, 6+t], linewidth=2, c=color[2], linestyle="-", label="ROT, training")
                plt.plot(range(1,101), tmp[:,21+t], linewidth=2, c=color[3], linestyle="-", label="CT, test")
                plt.plot(range(1,101), tmp[:,18+t], linewidth=2, c=color[4], linestyle="--", label="SMB, test")
                plt.plot(range(1,101), tmp[:,24+t], linewidth=2, c=color[5], linestyle="-", label="ROT, test")
            if t==0: plt.text(0.5, 0.95, '%s, On, %s, MZE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
            if t==1: plt.text(0.5, 0.95, '%s, On, %s, MAE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
            if t==2: plt.text(0.5, 0.95, '%s, On, %s, RMSE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
            plt.grid()
            plt.xlim(-1,102)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig("./on/%s-%s-ordered-%s.png"%(data,task,method),bbox_inches="tight", pad_inches=0.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()


#===== AD =====#
for d in range(len(D)):
    data = D[d]
    data2 = D2[d]
    for t in range(len(T)):
        task = T[t]
        method = 'AD'
        method2 = 'AD'
        fig=plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        tmp = np.loadtxt("../%s/threshold/%s-LC/training.log"%(data,method), delimiter=",")
        for i in range(100):
            if tmp[i,-(3-t)]==0:
                ax.axvspan(i+0.5, i+1.5, color="gray", edgecolor=None, alpha=0.5)
        plt.plot(range(1,101), tmp[:, 0+t], linewidth=2, c=color[6], linestyle="-", label="NN, training")
        plt.plot(range(1,101), tmp[:, 3+t], linewidth=2, c=color[2], linestyle="-", label="ROT, training")
        plt.plot(range(1,101), tmp[:,12+t], linewidth=2, c=color[7], linestyle="-", label="NN, test")
        plt.plot(range(1,101), tmp[:,15+t], linewidth=2, c=color[5], linestyle="-", label="ROT, test")
        if t==0: plt.text(0.5, 0.95, '%s, _, %s, MZE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
        if t==1: plt.text(0.5, 0.95, '%s, _, %s, MAE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
        if t==2: plt.text(0.5, 0.95, '%s, _, %s, RMSE'%(data2,method2),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes, fontsize=20)
        plt.grid()
        plt.xlim(-1,102)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("./%s-%s-%s.png"%(data,task,method),bbox_inches="tight", pad_inches=0.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
