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
TR, MP, NS, EP = 20, 5, 20, 200
datatype_set = ['DR5','DR10','OR']
dataname_set = [['abalone-5','bank1-5','bank2-5','calhousing-5','census1-5','census2-5','computer1-5','computer2-5'], 
                ['abalone-10','bank1-10','bank2-10','calhousing-10','census1-10','census2-10','computer1-10','computer2-10'], 
                ['car','ERA','LEV','SWD','winequality-red']]

#dataset
args = sys.argv
method, A, B = args[1], int(args[2]), int(args[3])
trte_data = np.loadtxt("../datasets/"+datatype_set[A]+"/"+dataname_set[A][B]+".csv", delimiter = ",")
samplenum = trte_data.shape[0]
dimension = trte_data.shape[1]-1
classnum  = int(np.max(trte_data[:,-1])-np.min(trte_data[:,-1])+1)
print(method, dataname_set[A][B], samplenum, dimension, classnum)

#learning function
def learning(seed, train_data, valid_data, test_data, node, epoch):
    #select devise
    torch.manual_seed(seed)
    device = torch.device('cpu')
    #arrange dataset
    train_loader, valid_loader, test_loader = MF.train_test_loader(train_data, valid_data, test_data, classnum, int(samplenum*0.72/10), "O", device)
    #set model, optimizer
    model = eval("MF."+method.split('_')[0])(dimension, node, classnum-1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3)
    #loop epoch
    if method.split('_')[1]=="POCLAT" or method.split('_')[1]=="POCLIT":
        res = np.zeros((epoch,39))
    else:
        res = np.zeros((epoch,30))
    for e in range(epoch):
        eval("MF.train_"+method.split('_')[1])(train_loader, classnum, device, model, optimizer)
        #
        if method.split('_')[1]=="POCLAT" or method.split('_')[1]=="POCLIT":
            t_Z, t_A, t_S = None, None, None
            res[e, 0: 3] = eval("MF.test_"+method.split('_')[1])(train_loader, classnum, "O", device, model, "RL", False, t_Z, t_A, t_S)
            res[e, 3: 6] = eval("MF.test_"+method.split('_')[1])(train_loader, classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
            res[e, 6: 9] = eval("MF.test_"+method.split('_')[1])(train_loader, classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
            res[e,12:15] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "RL", False, t_Z, t_A, t_S)
            res[e,15:18] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
            res[e,18:21] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
            res[e,24:27] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "RL", False, t_Z, t_A, t_S)
            res[e,27:30] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
            res[e,30:33] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
            #
            all_g, all_y = MF.B_all_gy(train_loader, device, model)
            t_Z=MF.IT_func(all_g,all_y,classnum,device,"Z"); t_A=MF.IT_func(all_g,all_y,classnum,device,"A"); t_S=MF.IT_func(all_g,all_y,classnum,device,"S")
            res[e, 9:12] = eval("MF.test_"+method.split('_')[1])(train_loader, classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
            res[e,21:24] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
            res[e,33:36] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
            #
            res[e,-3] = int(torch.equal(t_Z, torch.sort(t_Z)[0]))
            res[e,-2] = int(torch.equal(t_A, torch.sort(t_A)[0]))
            res[e,-1] = int(torch.equal(t_S, torch.sort(t_S)[0]))
            print(res[e,24:])
        else:
            t_Z, t_A, t_S = None, None, None
            res[e, 0: 3] = eval("MF.test_"+method.split('_')[1])(train_loader, classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
            res[e, 3: 6] = eval("MF.test_"+method.split('_')[1])(train_loader, classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
            res[e, 9:12] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
            res[e,12:15] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
            res[e,18:21] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
            res[e,21:24] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
            #
            all_g, all_y = MF.B_all_gy(train_loader, device, model)
            t_Z=MF.IT_func(all_g,all_y,classnum,device,"Z"); t_A=MF.IT_func(all_g,all_y,classnum,device,"A"); t_S=MF.IT_func(all_g,all_y,classnum,device,"S")
            res[e, 6: 9] = eval("MF.test_"+method.split('_')[1])(train_loader, classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
            res[e,15:18] = eval("MF.test_"+method.split('_')[1])(valid_loader, classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
            res[e,24:27] = eval("MF.test_"+method.split('_')[1])(test_loader,  classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
            #
            res[e,-3] = int(torch.equal(t_Z, torch.sort(t_Z)[0]))
            res[e,-2] = int(torch.equal(t_A, torch.sort(t_A)[0]))
            res[e,-1] = int(torch.equal(t_S, torch.sort(t_S)[0]))
            print(res[e,18:])
    return res

#test function
def test(seed):
    #load dataset
    trva_data,  test_data  = train_test_split(trte_data, test_size=0.2, random_state=seed, stratify=trte_data[:,-1])
    train_data, valid_data = train_test_split(trva_data, test_size=0.1, random_state=seed, stratify=trva_data[:,-1])
    #test
    res = learning(seed, train_data, valid_data, test_data, NS, EP)
    if os.path.isdir("./Results")==False: os.makedirs("./Results")
    np.savetxt("./Results/%s-%s.csv"%(method, dataname_set[A][B]), res, delimiter=",")

#main function
if __name__ == "__main__":
    #
    test(1)
