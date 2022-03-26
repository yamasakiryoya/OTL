import re
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

methods  = ['SVOR','NLL','ANLCL','ordered-SVOR','ordered-NLL','ordered-ANLCL','AD']
datasets = ['morph2','cacd','afad']
TNUM = 20
RES_A, MEA_A, STD_A = np.zeros((3,18,TNUM)), np.zeros((3,18)), np.zeros((3,18))
RES_S, MEA_S, STD_S = np.zeros((3,18,TNUM)), np.zeros((3,18)), np.zeros((3,18))

for M in [0,1,2,3,4,5,6]:
    m = methods[M]
    for D in [0,1,2]:
        d = datasets[D]
        for seed in range(TNUM):
            path = '../' + d + '/MTM/' + m + '/seed' + str(seed) + '/training.log'
            with open(path) as f: l1 = f.readlines()
            #
            if M==0:
                RES_A[D, 0,seed] = re.split('[/ ]', re.split('[,]', l1[-5])[-2])[-2]
                RES_S[D, 0,seed] = re.split('[/ ]', re.split('[,]', l1[-4])[-2])[-1]
                RES_A[D, 1,seed] = re.split('[/ ]', re.split('[,]', l1[-2])[-2])[-2]
                RES_S[D, 1,seed] = re.split('[/ ]', re.split('[,]', l1[-1])[-2])[-1]
            if M==1:
                RES_A[D, 2,seed] = re.split('[/ ]', re.split('[,]', l1[-5])[-2])[-2]
                RES_S[D, 2,seed] = re.split('[/ ]', re.split('[,]', l1[-4])[-2])[-1]
                RES_A[D, 3,seed] = re.split('[/ ]', l1[-8])[-2]
                RES_S[D, 3,seed] = re.split('[/ ]', l1[-7])[-1]
                RES_A[D, 4,seed] = re.split('[/ ]', re.split('[,]', l1[-2])[-2])[-2]
                RES_S[D, 4,seed] = re.split('[/ ]', re.split('[,]', l1[-1])[-2])[-1]
            if M==2:
                RES_A[D, 5,seed] = re.split('[/ ]', re.split('[,]', l1[-5])[-2])[-2]
                RES_S[D, 5,seed] = re.split('[/ ]', re.split('[,]', l1[-4])[-2])[-1]
                RES_A[D, 6,seed] = re.split('[/ ]', l1[-8])[-2]
                RES_S[D, 6,seed] = re.split('[/ ]', l1[-7])[-1]
                RES_A[D, 7,seed] = re.split('[/ ]', re.split('[,]', l1[-2])[-2])[-2]
                RES_S[D, 7,seed] = re.split('[/ ]', re.split('[,]', l1[-1])[-2])[-1]
            if M==3:
                RES_A[D, 8,seed] = re.split('[/ ]', re.split('[,]', l1[-5])[-2])[-2]
                RES_S[D, 8,seed] = re.split('[/ ]', re.split('[,]', l1[-4])[-2])[-1]
                RES_A[D, 9,seed] = re.split('[/ ]', re.split('[,]', l1[-2])[-2])[-2]
                RES_S[D, 9,seed] = re.split('[/ ]', re.split('[,]', l1[-1])[-2])[-1]
            if M==4:
                RES_A[D,10,seed] = re.split('[/ ]', re.split('[,]', l1[-5])[-2])[-2]
                RES_S[D,10,seed] = re.split('[/ ]', re.split('[,]', l1[-4])[-2])[-1]
                RES_A[D,11,seed] = re.split('[/ ]', l1[-8])[-2]
                RES_S[D,11,seed] = re.split('[/ ]', l1[-7])[-1]
                RES_A[D,12,seed] = re.split('[/ ]', re.split('[,]', l1[-2])[-2])[-2]
                RES_S[D,12,seed] = re.split('[/ ]', re.split('[,]', l1[-1])[-2])[-1]
            if M==5:
                RES_A[D,13,seed] = re.split('[/ ]', re.split('[,]', l1[-5])[-2])[-2]
                RES_S[D,13,seed] = re.split('[/ ]', re.split('[,]', l1[-4])[-2])[-1]
                RES_A[D,14,seed] = re.split('[/ ]', l1[-8])[-2]
                RES_S[D,14,seed] = re.split('[/ ]', l1[-7])[-1]
                RES_A[D,15,seed] = re.split('[/ ]', re.split('[,]', l1[-2])[-2])[-2]
                RES_S[D,15,seed] = re.split('[/ ]', re.split('[,]', l1[-1])[-2])[-1]
            if M==6:
                RES_A[D,16,seed] = re.split('[/ ]', l1[-5])[-2]
                RES_S[D,16,seed] = re.split('[/ ]', l1[-4])[-1]
                RES_A[D,17,seed] = re.split('[/ ]', re.split('[,]', l1[-2])[-2])[-2]
                RES_S[D,17,seed] = re.split('[/ ]', re.split('[,]', l1[-1])[-2])[-1]


for M in range(18):
    for D in [0,1,2]:
        MEA_A[D,M] = np.mean(RES_A[D,M,:])
        MEA_S[D,M] = np.mean(RES_S[D,M,:])
        STD_A[D,M] = np.std(RES_A[D,M,:])
        STD_S[D,M] = np.std(RES_S[D,M,:])

for M in range(18):
    for D in [0,1,2]:
        print("$%.3f\pm%.3f$ & "%(
            Decimal(str(MEA_A[D,M])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            Decimal(str(STD_A[D,M])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),end='')
        print("$%.3f\pm%.3f$ & "%(
            Decimal(str(MEA_S[D,M])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            Decimal(str(STD_S[D,M])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),end='')
    print("")