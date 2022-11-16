#import
import numpy as np
from scipy import stats

ml1 = [18,27,18,27,27]
ml2 = [[0,1],[0,1,2],[0,1],[0,1,2],[0,1,2]]

MEA = np.zeros((3,5,13))
RES = np.zeros((3,5,13,50))

#MAE_{STD}
for t in range(3):
	j=0
	for m in range(5):
		me = ["AD","ODB_LogiIT","ODOB_LogiIT","ODOB_LogiAT","POOCL_NLL"][m]
		for i in ml2[m]:
			for d in range(5):
				de = ["LEV","ERA","SWD","winequality-red","car"][d]
				tmp = np.loadtxt("./Results/%s-%s.csv"%(me,de), delimiter=",")
				print("$%.3f_{%.3f}$"%(np.mean(tmp[:50,int(ml1[m]/3*2+3*i+t)]),np.std(tmp[:50,int(ml1[m]/3*2+3*i+t)])), end=" & ")
				MEA[t,d,j] = np.mean(tmp[:50,int(ml1[m]/3*2+3*i+t)])
				RES[t,d,j,:] = tmp[:50,int(ml1[m]/3*2+3*i+t)]
			print("")
			j+=1
		print("")
	print("")

#TOP
for t in range(3):
	for d in range(5):
		for i in range(13):
			if np.min(MEA[t,d,:])==MEA[t,d,i]:
				print(i, end=" ")
		print("")
	print("")

#TEST
for t in range(3):
	for d in range(5):
		print(t,d,np.argmin(np.array([MEA[t,d,0],MEA[t,d,1]])),"  %.3f"%
			stats.mannwhitneyu(RES[t,d,0,:], RES[t,d,1,:], alternative='greater').pvalue)
	print("")
	for d in range(5):
		print(t,d,np.argmin(np.array([MEA[t,d,2],MEA[t,d,3],MEA[t,d,4]])),"  %.3f  %.3f  %.3f"%(
			stats.mannwhitneyu(RES[t,d,2,:], RES[t,d,3,:], alternative='greater').pvalue,
			stats.mannwhitneyu(RES[t,d,2,:], RES[t,d,4,:], alternative='greater').pvalue,
			stats.mannwhitneyu(RES[t,d,3,:], RES[t,d,4,:], alternative='greater').pvalue))
	print("")
	for d in range(5):
		print(t,d,np.argmin(np.array([MEA[t,d,5],MEA[t,d,6]])),"  %.3f"%
			stats.mannwhitneyu(RES[t,d,5,:], RES[t,d,6,:], alternative='greater').pvalue)
	print("")
	for d in range(5):
		print(t,d,np.argmin(np.array([MEA[t,d,7],MEA[t,d,8],MEA[t,d,9]])),"  %.3f  %.3f  %.3f"%(
		stats.mannwhitneyu(RES[t,d,7,:], RES[t,d,8,:], alternative='greater').pvalue,
		stats.mannwhitneyu(RES[t,d,7,:], RES[t,d,9,:], alternative='greater').pvalue,
		stats.mannwhitneyu(RES[t,d,8,:], RES[t,d,9,:], alternative='greater').pvalue))
	print("")
	for d in range(5):
		print(t,d,np.argmin(np.array([MEA[t,d,10],MEA[t,d,11],MEA[t,d,12]])),"  %.3f  %.3f  %.3f"%(
		stats.mannwhitneyu(RES[t,d,10,:], RES[t,d,11,:], alternative='greater').pvalue,
		stats.mannwhitneyu(RES[t,d,10,:], RES[t,d,12,:], alternative='greater').pvalue,
		stats.mannwhitneyu(RES[t,d,11,:], RES[t,d,12,:], alternative='greater').pvalue))
