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
D = ['contact-lenses','pasture','squash-stored','squash-unstored','bondrate','tae','automobile','newthyroid','toy','ESL','balance-scale','eucalyptus','LEV','ERA','SWD','winequality-red','car']
#D = ['EF3-diabetes','EF3-pyrimidines','EF3-auto-price','EF3-servo','EF3-triazines','EF3-wisconsin-breast-cancer','EF3-machine-cpu','EF3-auto-mpg','EF3-boston-housing','EF3-stock-domain','EF3-abalone','EF3-delta-ailerons','EF3-kinematics-of-robot-arm','EF3-computer-activity-1','EF3-pumadyn-domain-1','EF3-bank-domain-1','EF3-computer-activity-2','EF3-pumadyn-domain-2','EF3-bank-domain-2','EF3-delta-elevators','EF3-pole-telecomm','EF3-ailerons','EF3-elevators','EF3-california-housing','EF3-census-1','EF3-census-2','EF3-2d-planes','EF3-friedman-artificial','EF3-mv-artificial']
#D = ['EF5-diabetes','EF5-pyrimidines','EF5-auto-price','EF5-servo','EF5-triazines','EF5-wisconsin-breast-cancer','EF5-machine-cpu','EF5-auto-mpg','EF5-boston-housing','EF5-stock-domain','EF5-abalone','EF5-delta-ailerons','EF5-kinematics-of-robot-arm','EF5-computer-activity-1','EF5-pumadyn-domain-1','EF5-bank-domain-1','EF5-computer-activity-2','EF5-pumadyn-domain-2','EF5-bank-domain-2','EF5-delta-elevators','EF5-pole-telecomm','EF5-ailerons','EF5-elevators','EF5-california-housing','EF5-census-1','EF5-census-2','EF5-2d-planes','EF5-friedman-artificial','EF5-mv-artificial']
#D = ['EF10-diabetes','EF10-pyrimidines','EF10-auto-price','EF10-servo','EF10-triazines','EF10-wisconsin-breast-cancer','EF10-machine-cpu','EF10-auto-mpg','EF10-boston-housing','EF10-stock-domain','EF10-abalone','EF10-delta-ailerons','EF10-kinematics-of-robot-arm','EF10-computer-activity-1','EF10-pumadyn-domain-1','EF10-bank-domain-1','EF10-computer-activity-2','EF10-pumadyn-domain-2','EF10-bank-domain-2','EF10-delta-elevators','EF10-pole-telecomm','EF10-ailerons','EF10-elevators','EF10-california-housing','EF10-census-1','EF10-census-2','EF10-2d-planes','EF10-friedman-artificial','EF10-mv-artificial']
#D = ['EL3-diabetes','EL3-pyrimidines','EL3-auto-price','EL3-servo','EL3-triazines','EL3-wisconsin-breast-cancer','EL3-machine-cpu','EL3-auto-mpg','EL3-boston-housing','EL3-stock-domain','EL3-abalone','EL3-delta-ailerons','EL3-kinematics-of-robot-arm','EL3-computer-activity-1','EL3-pumadyn-domain-1','EL3-bank-domain-1','EL3-computer-activity-2','EL3-pumadyn-domain-2','EL3-bank-domain-2','EL3-delta-elevators','EL3-pole-telecomm','EL3-ailerons','EL3-elevators','EL3-california-housing','EL3-census-1','EL3-census-2','EL3-2d-planes','EL3-friedman-artificial','EL3-mv-artificial']
#D = ['EL5-diabetes','EL5-pyrimidines','EL5-auto-price','EL5-servo','EL5-triazines','EL5-wisconsin-breast-cancer','EL5-machine-cpu','EL5-auto-mpg','EL5-boston-housing','EL5-stock-domain','EL5-abalone','EL5-delta-ailerons','EL5-kinematics-of-robot-arm','EL5-computer-activity-1','EL5-pumadyn-domain-1','EL5-bank-domain-1','EL5-computer-activity-2','EL5-pumadyn-domain-2','EL5-bank-domain-2','EL5-delta-elevators','EL5-pole-telecomm','EL5-ailerons','EL5-elevators','EL5-california-housing','EL5-census-1','EL5-census-2','EL5-2d-planes','EL5-friedman-artificial','EL5-mv-artificial']
#D = ['EL10-diabetes','EL10-pyrimidines','EL10-auto-price','EL10-servo','EL10-triazines','EL10-wisconsin-breast-cancer','EL10-machine-cpu','EL10-auto-mpg','EL10-boston-housing','EL10-stock-domain','EL10-abalone','EL10-delta-ailerons','EL10-kinematics-of-robot-arm','EL10-computer-activity-1','EL10-pumadyn-domain-1','EL10-bank-domain-1','EL10-computer-activity-2','EL10-pumadyn-domain-2','EL10-bank-domain-2','EL10-delta-elevators','EL10-pole-telecomm','EL10-ailerons','EL10-elevators','EL10-california-housing','EL10-census-1','EL10-census-2','EL10-2d-planes','EL10-friedman-artificial','EL10-mv-artificial']
M  = ["AD","ODB_HingIT","ODOB_HingIT","ODOB_HingAT","ODB_LogiIT","ODOB_LogiIT","ODOB_LogiAT","POOCL_NLL"]
M2 = ["AD","HingIT, non-ordered","HingIT, ordered","HingAT, ordered","LogiIT, non-ordered","LogiIT, ordered","LogiAT, ordered","OLR_NLL, ordered"]

for d in range(len(D)):
    data = D[d]
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
                elif m==2 or m==3 or m==5:
                    plt.plot(range(1,501), tmp[:, 0+t], linewidth=2, c=color[1], label="MT, ST, training")
                    plt.plot(range(1,501), tmp[:, 3+t], linewidth=2, c=color[4], label="EOT, training")
                    plt.plot(range(1,501), tmp[:,12+t], linewidth=2, c=color[1], linestyle="--", label="MT, ST, test")
                    plt.plot(range(1,501), tmp[:,15+t], linewidth=2, c=color[4], linestyle="--", label="EOT, test")
                    Mi = np.min([np.min(tmp[5:, 0+t]),np.min(tmp[5:, 3+t]),np.min(tmp[5:,12+t]),np.min(tmp[5:,15+t])])
                    Ma = np.max([np.max(tmp[5:, 0+t]),np.max(tmp[5:, 3+t]),np.max(tmp[5:,12+t]),np.max(tmp[5:,15+t])])
                elif m==6 or m==7:
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
