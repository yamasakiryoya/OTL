import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'k'

color = ['r','g','b','c','m']
T  = ['Z','A','S']
D  = ['morph2','cacd','afad']
M  = ['AD','hing-IT','ord-hing-IT','ord-hing-AT','logi-IT','ord-logi-IT','ord-logi-AT','POOCL-NLL']
L  = [['NT','OT'],['MT','ST','OT'],['ST','OT'],['ST','OT'],['MT','ST','OT'],['ST','OT'],['ST','RL','OT'],['ST','RL','OT']]
Ls = ['NT','MT','ST','RL','OT']
if os.path.isdir("./figure")==False: os.makedirs("./figure")

for t in range(len(T)):
    for d in D:
        for m in range(len(M)):
            fig = plt.figure(figsize=(15,6)); ax = fig.add_subplot(111)
            for l in L[m]:
                tmp = np.loadtxt("../%s/result/%s/seed0/training_%s.log"%(d,M[m],l), delimiter=",")
                if t==2: tmp = np.sqrt(tmp)
                ax.plot(range(1,101), tmp[:,  t], linewidth=2, c=color[Ls.index(l)], linestyle="-",  label="tra. %s"%l)
                ax.plot(range(1,101), tmp[:,6+t], linewidth=2, c=color[Ls.index(l)], linestyle="--", label="tes. %s"%l)
                #
                print("%s, %s-%s, %s, tra. %.d-%.4f, tes. %.d-%.4f"%(d,M[m],l,T[t],np.argmin(tmp[:,t]),np.min(tmp[:,t]),np.argmin(tmp[:,6+t]),np.min(tmp[:,6+t])))
                #
            plt.xlim(-1,102)
            plt.grid()
            plt.tight_layout()
            plt.savefig("./figure/%s-%s-%s.png"%(d,M[m],T[t]),bbox_inches="tight", pad_inches=0.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()