#import
import numpy as np
from scipy import stats

ml1 = [18,27,18,18,27,18,27,27]
ml2 = [[0,1],[0,1,2],[0,1],[0,1],[0,1,2],[0,1],[0,1,2],[0,1,2]]

methods  = ["AD","ODB_HingIT","ODOB_HingIT","ODOB_HingAT","ODB_LogiIT","ODOB_LogiIT","ODOB_LogiAT","POOCL_NLL"]
datasets2= [['contact-lenses','pasture','squash-stored','squash-unstored','bondrate','tae','automobile','newthyroid','toy','ESL','balance-scale','eucalyptus','LEV','ERA','SWD','winequality-red','car'],
			['EF3-diabetes','EF3-pyrimidines','EF3-auto-price','EF3-servo','EF3-triazines','EF3-wisconsin-breast-cancer','EF3-machine-cpu','EF3-auto-mpg','EF3-boston-housing','EF3-stock-domain','EF3-abalone','EF3-delta-ailerons','EF3-kinematics-of-robot-arm','EF3-computer-activity-1','EF3-pumadyn-domain-1','EF3-bank-domain-1','EF3-computer-activity-2','EF3-pumadyn-domain-2','EF3-bank-domain-2','EF3-delta-elevators','EF3-pole-telecomm','EF3-ailerons','EF3-elevators','EF3-california-housing','EF3-census-1','EF3-census-2','EF3-2d-planes','EF3-friedman-artificial','EF3-mv-artificial'],
			['EF5-diabetes','EF5-pyrimidines','EF5-auto-price','EF5-servo','EF5-triazines','EF5-wisconsin-breast-cancer','EF5-machine-cpu','EF5-auto-mpg','EF5-boston-housing','EF5-stock-domain','EF5-abalone','EF5-delta-ailerons','EF5-kinematics-of-robot-arm','EF5-computer-activity-1','EF5-pumadyn-domain-1','EF5-bank-domain-1','EF5-computer-activity-2','EF5-pumadyn-domain-2','EF5-bank-domain-2','EF5-delta-elevators','EF5-pole-telecomm','EF5-ailerons','EF5-elevators','EF5-california-housing','EF5-census-1','EF5-census-2','EF5-2d-planes','EF5-friedman-artificial','EF5-mv-artificial'],
			['EF10-diabetes','EF10-pyrimidines','EF10-auto-price','EF10-servo','EF10-triazines','EF10-wisconsin-breast-cancer','EF10-machine-cpu','EF10-auto-mpg','EF10-boston-housing','EF10-stock-domain','EF10-abalone','EF10-delta-ailerons','EF10-kinematics-of-robot-arm','EF10-computer-activity-1','EF10-pumadyn-domain-1','EF10-bank-domain-1','EF10-computer-activity-2','EF10-pumadyn-domain-2','EF10-bank-domain-2','EF10-delta-elevators','EF10-pole-telecomm','EF10-ailerons','EF10-elevators','EF10-california-housing','EF10-census-1','EF10-census-2','EF10-2d-planes','EF10-friedman-artificial','EF10-mv-artificial'],
			['EL3-diabetes','EL3-pyrimidines','EL3-auto-price','EL3-servo','EL3-triazines','EL3-wisconsin-breast-cancer','EL3-machine-cpu','EL3-auto-mpg','EL3-boston-housing','EL3-stock-domain','EL3-abalone','EL3-delta-ailerons','EL3-kinematics-of-robot-arm','EL3-computer-activity-1','EL3-pumadyn-domain-1','EL3-bank-domain-1','EL3-computer-activity-2','EL3-pumadyn-domain-2','EL3-bank-domain-2','EL3-delta-elevators','EL3-pole-telecomm','EL3-ailerons','EL3-elevators','EL3-california-housing','EL3-census-1','EL3-census-2','EL3-2d-planes','EL3-friedman-artificial','EL3-mv-artificial'],
			['EL5-diabetes','EL5-pyrimidines','EL5-auto-price','EL5-servo','EL5-triazines','EL5-wisconsin-breast-cancer','EL5-machine-cpu','EL5-auto-mpg','EL5-boston-housing','EL5-stock-domain','EL5-abalone','EL5-delta-ailerons','EL5-kinematics-of-robot-arm','EL5-computer-activity-1','EL5-pumadyn-domain-1','EL5-bank-domain-1','EL5-computer-activity-2','EL5-pumadyn-domain-2','EL5-bank-domain-2','EL5-delta-elevators','EL5-pole-telecomm','EL5-ailerons','EL5-elevators','EL5-california-housing','EL5-census-1','EL5-census-2','EL5-2d-planes','EL5-friedman-artificial','EL5-mv-artificial'],
			['EL10-diabetes','EL10-pyrimidines','EL10-auto-price','EL10-servo','EL10-triazines','EL10-wisconsin-breast-cancer','EL10-machine-cpu','EL10-auto-mpg','EL10-boston-housing','EL10-stock-domain','EL10-abalone','EL10-delta-ailerons','EL10-kinematics-of-robot-arm','EL10-computer-activity-1','EL10-pumadyn-domain-1','EL10-bank-domain-1','EL10-computer-activity-2','EL10-pumadyn-domain-2','EL10-bank-domain-2','EL10-delta-elevators','EL10-pole-telecomm','EL10-ailerons','EL10-elevators','EL10-california-housing','EL10-census-1','EL10-census-2','EL10-2d-planes','EL10-friedman-artificial','EL10-mv-artificial']]

FIN = np.zeros((3,20))
for D in range(7):
	datasets = datasets2[D]
	MEA = np.zeros((3,len(datasets),20))
	RET = np.zeros((3,20))
	#MAE_{STD}
	for t in range(3):
		j=0
		for m in range(len(methods)):
			me = methods[m]
			for i in ml2[m]:
				for d in range(len(datasets)):
					de = datasets[d]
					tmp = np.loadtxt("./Results/%s-%s.csv"%(me,de), delimiter=",")
					if t==2: tmp = np.sqrt(tmp)
					MEA[t,d,j] = np.mean(tmp[:50,int(ml1[m]/3*2+3*i+t)])
				j+=1

	#TOP
	for t in range(3):
		for d in range(len(datasets)):
			for i in range(20):
				if np.min(MEA[t,d,:])==MEA[t,d,i]:
					RET[t,i] += 1


	for i in range(20):
		print("%d,%d,%d &"%(RET[0,i],RET[1,i],RET[2,i]))
		FIN[0,i] += RET[0,i]
		FIN[1,i] += RET[1,i]
		FIN[2,i] += RET[2,i]
	print("\n")

for i in range(20):
	print("%d,%d,%d \\\\"%(FIN[0,i],FIN[1,i],FIN[2,i]))