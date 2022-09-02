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


def learning(seed, train_data, valid_data, test_data, node, epoch):
    #select devise
    torch.manual_seed(seed)
    device = torch.device('cpu')
    #arrange dataset
    train_loader, valid_loader, test_loader = MF.train_test_loader(train_data, valid_data, test_data, classnum, int(samplenum*0.72/10), "O", device)
    #set model, optimizer
    if method=="POOCL_NLL":
        model = MF.ODOB(dimension, node, classnum-1).to(device)
    else:
        model = MF.ODB(dimension, node, classnum-1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.1**3)
    #loop epoch
    res = np.zeros((epoch,39))
    for e in range(epoch):
        eval("MF.train_"+method)(train_loader, classnum, device, model, optimizer)
        #
        t_Z, t_A, t_S = None, None, None
        res[e, 0: 3] = eval("MF.test_"+method)(train_loader, classnum, "O", device, model, "RL", False, t_Z, t_A, t_S)
        res[e, 3: 6] = eval("MF.test_"+method)(train_loader, classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
        res[e, 6: 9] = eval("MF.test_"+method)(train_loader, classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
        res[e,12:15] = eval("MF.test_"+method)(valid_loader, classnum, "O", device, model, "RL", False, t_Z, t_A, t_S)
        res[e,15:18] = eval("MF.test_"+method)(valid_loader, classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
        res[e,18:21] = eval("MF.test_"+method)(valid_loader, classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
        res[e,24:27] = eval("MF.test_"+method)(test_loader,  classnum, "O", device, model, "RL", False, t_Z, t_A, t_S)
        res[e,27:30] = eval("MF.test_"+method)(test_loader,  classnum, "O", device, model, "MT", False, t_Z, t_A, t_S)
        res[e,30:33] = eval("MF.test_"+method)(test_loader,  classnum, "O", device, model, "ST", False, t_Z, t_A, t_S)
        #
        all_g, all_y = MF.B_all_gy(train_loader, device, model)
        t_Z=MF.IT_func(all_g,all_y,classnum,device,"Z"); t_A=MF.IT_func(all_g,all_y,classnum,device,"A"); t_S=MF.IT_func(all_g,all_y,classnum,device,"S")
        res[e, 9:12] = eval("MF.test_"+method)(train_loader, classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
        res[e,21:24] = eval("MF.test_"+method)(valid_loader, classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
        res[e,33:36] = eval("MF.test_"+method)(test_loader,  classnum, "O", device, model, "IT", False, t_Z, t_A, t_S)
        #
        res[e,-3] = int(torch.equal(t_Z, torch.sort(t_Z)[0]))
        res[e,-2] = int(torch.equal(t_A, torch.sort(t_A)[0]))
        res[e,-1] = int(torch.equal(t_S, torch.sort(t_S)[0]))
        print(res[e,24:])
    result = np.zeros(len(res[0,:]))
    for i in range((len(res[0,:])-3)):
        result[i] = res[MF.back_argmin(res[:,int(((len(res[0,:])-3)/3)+i%((len(res[0,:])-3)/3))]),i]
    result[-3] = res[MF.back_argmin(res[:,int(((len(res[0,:])-3)/3*2)-2)]),-3]
    result[-2] = res[MF.back_argmin(res[:,int(((len(res[0,:])-3)/3*2)-1)]),-2]
    result[-1] = res[MF.back_argmin(res[:,int(((len(res[0,:])-3)/3*2))  ]),-1]
    return result

#test function
def test(seed, results):
    #load dataset
    trva_data,  test_data  = train_test_split(trte_data, test_size=0.2, random_state=seed, stratify=trte_data[:,-1])
    train_data, valid_data = train_test_split(trva_data, test_size=0.1, random_state=seed, stratify=trva_data[:,-1])
    #test
    res = learning(seed, train_data, valid_data, test_data, NS, EP)
    results[seed] = res.tolist()

#parallel processing
def parallel():
    manager = Manager()
    jobs, results = [], manager.list(range(TR))
    for i in range(int(TR//MP)):
        for seed in range(i*MP,(i+1)*MP):
            job = Process(target=test, args=(seed, results))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()
    tmp = np.array(results)
    res = np.zeros((int(TR+2), tmp.shape[1]))
    res[:TR,:], res[TR,:], res[int(TR+1),:] = tmp, mean(tmp,axis=0), std(tmp,axis=0)
    if os.path.isdir("./Results")==False: os.makedirs("./Results")
    np.savetxt("./Results/%s-%s.csv"%(method, dataname_set[A][B]), res, delimiter=",")

#main function
if __name__ == "__main__":
    #
    parallel()