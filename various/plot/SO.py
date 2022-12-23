#import
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import mean,std
from sklearn.model_selection import train_test_split
from multiprocessing import Pool,Process,Pipe,Manager
import MyFunc as MF

#parameter
TR, MP, NS = 50, 5, 100
datatype_set = ['OR','EF3','EF5','EF10','EL3','EL5','EL10']
dataname_set = [['contact-lenses','pasture','squash-stored','squash-unstored','tae','newthyroid','balance-scale','SWD','car','bondrate','toy','eucalyptus','LEV','automobile','winequality-red','ESL','ERA'],
                ['EF3-2d-planes','EF3-abalone','EF3-ailerons','EF3-auto-mpg','EF3-auto-price','EF3-bank-domain-1','EF3-bank-domain-2','EF3-boston-housing','EF3-california-housing','EF3-census-1','EF3-census-2','EF3-computer-activity-1','EF3-computer-activity-2','EF3-delta-ailerons','EF3-delta-elevators','EF3-diabetes','EF3-elevators','EF3-friedman-artificial','EF3-kinematics-of-robot-arm','EF3-machine-cpu','EF3-mv-artificial','EF3-pole-telecomm','EF3-pumadyn-domain-1','EF3-pumadyn-domain-2','EF3-pyrimidines','EF3-servo','EF3-stock-domain','EF3-triazines','EF3-wisconsin-breast-cancer'], 
                ['EF5-2d-planes','EF5-abalone','EF5-ailerons','EF5-auto-mpg','EF5-auto-price','EF5-bank-domain-1','EF5-bank-domain-2','EF5-boston-housing','EF5-california-housing','EF5-census-1','EF5-census-2','EF5-computer-activity-1','EF5-computer-activity-2','EF5-delta-ailerons','EF5-delta-elevators','EF5-diabetes','EF5-elevators','EF5-friedman-artificial','EF5-kinematics-of-robot-arm','EF5-machine-cpu','EF5-mv-artificial','EF5-pole-telecomm','EF5-pumadyn-domain-1','EF5-pumadyn-domain-2','EF5-pyrimidines','EF5-servo','EF5-stock-domain','EF5-triazines','EF5-wisconsin-breast-cancer'], 
                ['EF10-2d-planes','EF10-abalone','EF10-ailerons','EF10-auto-mpg','EF10-auto-price','EF10-bank-domain-1','EF10-bank-domain-2','EF10-boston-housing','EF10-california-housing','EF10-census-1','EF10-census-2','EF10-computer-activity-1','EF10-computer-activity-2','EF10-delta-ailerons','EF10-delta-elevators','EF10-diabetes','EF10-elevators','EF10-friedman-artificial','EF10-kinematics-of-robot-arm','EF10-machine-cpu','EF10-mv-artificial','EF10-pole-telecomm','EF10-pumadyn-domain-1','EF10-pumadyn-domain-2','EF10-pyrimidines','EF10-servo','EF10-stock-domain','EF10-triazines','EF10-wisconsin-breast-cancer'], 
                ['EL3-2d-planes','EL3-abalone','EL3-ailerons','EL3-auto-mpg','EL3-auto-price','EL3-bank-domain-1','EL3-bank-domain-2','EL3-boston-housing','EL3-california-housing','EL3-census-1','EL3-census-2','EL3-computer-activity-1','EL3-computer-activity-2','EL3-delta-ailerons','EL3-delta-elevators','EL3-diabetes','EL3-elevators','EL3-friedman-artificial','EL3-kinematics-of-robot-arm','EL3-machine-cpu','EL3-mv-artificial','EL3-pole-telecomm','EL3-pumadyn-domain-1','EL3-pumadyn-domain-2','EL3-pyrimidines','EL3-servo','EL3-stock-domain','EL3-triazines','EL3-wisconsin-breast-cancer'], 
                ['EL5-2d-planes','EL5-abalone','EL5-ailerons','EL5-auto-mpg','EL5-auto-price','EL5-bank-domain-1','EL5-bank-domain-2','EL5-boston-housing','EL5-california-housing','EL5-census-1','EL5-census-2','EL5-computer-activity-1','EL5-computer-activity-2','EL5-delta-ailerons','EL5-delta-elevators','EL5-diabetes','EL5-elevators','EL5-friedman-artificial','EL5-kinematics-of-robot-arm','EL5-machine-cpu','EL5-mv-artificial','EL5-pole-telecomm','EL5-pumadyn-domain-1','EL5-pumadyn-domain-2','EL5-pyrimidines','EL5-servo','EL5-stock-domain','EL5-triazines','EL5-wisconsin-breast-cancer'], 
                ['EL10-2d-planes','EL10-abalone','EL10-ailerons','EL10-auto-mpg','EL10-auto-price','EL10-bank-domain-1','EL10-bank-domain-2','EL10-boston-housing','EL10-california-housing','EL10-census-1','EL10-census-2','EL10-computer-activity-1','EL10-computer-activity-2','EL10-delta-ailerons','EL10-delta-elevators','EL10-diabetes','EL10-elevators','EL10-friedman-artificial','EL10-kinematics-of-robot-arm','EL10-machine-cpu','EL10-mv-artificial','EL10-pole-telecomm','EL10-pumadyn-domain-1','EL10-pumadyn-domain-2','EL10-pyrimidines','EL10-servo','EL10-stock-domain','EL10-triazines','EL10-wisconsin-breast-cancer']]

#dataset
args = sys.argv
method, A, B = args[1], int(args[2]), int(args[3])
trte_data = np.loadtxt("../datasets/"+datatype_set[A]+"/"+dataname_set[A][B]+".csv", delimiter = ",")
samplenum = trte_data.shape[0]
dimension = trte_data.shape[1]-1
classnum  = int(np.max(trte_data[:,-1])-np.min(trte_data[:,-1])+1)
if samplenum<=2000: EP = 500
else: EP = 100
print(method, dataname_set[A][B], samplenum, dimension, classnum)

#learning function
def learning(seed, train_data, valid_data, test_data, node, epoch):
    #select devise
    torch.manual_seed(seed)
    device = torch.device('cpu')
    #arrange dataset
    if train_data.shape[0]<256: train_loader, train_loader2, valid_loader, test_loader = MF.train_test_loader(train_data, valid_data, test_data, classnum, 16, "O", device)
    else: train_loader, train_loader2, valid_loader, test_loader = MF.train_test_loader(train_data, valid_data, test_data, classnum, 256, "O", device)
    #set model, optimizer
    model = MF.ODOB(dimension, node, classnum-1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**2.5, eps=.1**6)
    #loop epoch
    res = np.zeros((epoch,18))
    for e in range(epoch):
        eval("MF.train_"+method.split('_')[1])(train_loader, classnum, device, model, optimizer)
        #
        a,y = MF.B_all_gy(train_loader2, device, model)
        t_Z=MF.OT_func(a,y,classnum,device,"Z"); t_A=MF.OT_func(a,y,classnum,device,"A"); t_S=MF.OT_func(a,y,classnum,device,"S")
        res[e, 0: 3] = eval("MF.test_"+method.split('_')[1])(train_loader2, classnum, "O", device, model, "ST", False, None, None, None)
        res[e, 3: 6] = eval("MF.test_"+method.split('_')[1])(train_loader2, classnum, "O", device, model, "OT", False, t_Z, t_A, t_S)
        #
        res[e, 6: 9] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "ST", False, None, None, None)
        res[e, 9:12] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "OT", False, t_Z, t_A, t_S)
        #
        res[e,12:15] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "ST", False, None, None, None)
        res[e,15:18] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "OT", False, t_Z, t_A, t_S)
        #print(res[e,12:])
    return res

#test function
def test(seed):
    #load dataset
    train_data, vate_data = train_test_split(trte_data, train_size=0.72, random_state=seed)
    valid_data, test_data = train_test_split(vate_data, train_size=0.08/0.28, random_state=seed)
    #test
    res = learning(seed, train_data, valid_data, test_data, NS, EP)
    if os.path.isdir("./Results")==False: os.makedirs("./Results")
    np.savetxt("./Results/%s-%s.csv"%(method, dataname_set[A][B]), res, delimiter=",")

#main function
if __name__ == "__main__":
    #
    test(0)
