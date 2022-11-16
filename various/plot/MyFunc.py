#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import mean,std,exp,sqrt
from itertools import product
from scipy.special import comb


#====================  tool  ====================#
def L_mean(arr, L):
    n = len(arr)
    tmp = np.zeros(n)
    for i in range(n):
        l = np.max([0,i-L])
        r = np.min([n,i+L+1])
        tmp[i] = np.mean(np.nan_to_num(arr[l:r], nan=10.**8))
    return tmp
def back_argmin(arr):
    #argmin
    return len(arr)-1-np.argmin(np.nan_to_num(arr[::-1], nan=10.**8))


#====================  zigzag  ====================#
arr3  = [0,2,1]
arr4  = [0,3,1,2]
arr5  = [0,4,1,3,2]
arr6  = [0,5,1,4,2,3]
arr9  = [0,8,1,7,2,6,3,5,4]
arr10 = [0,9,1,8,2,7,3,6,4,5]
def zigzag(arr, K, device):
    n = len(arr)
    res = torch.zeros(n, dtype=torch.long).to(device)
    if K == 3:
        for i in range(n):
            res[i] = arr3[int(arr[i])]
    elif K == 4:
        for i in range(n):
            res[i] = arr4[int(arr[i])]
    elif K == 5:
        for i in range(n):
            res[i] = arr5[int(arr[i])]
    elif K == 6:
        for i in range(n):
            res[i] = arr6[int(arr[i])]
    elif K == 9:
        for i in range(n):
            res[i] = arr9[int(arr[i])]
    elif K == 10:
        for i in range(n):
            res[i] = arr10[int(arr[i])]
    return res
def zigzag_inv(arr, K, device):
    n = len(arr)
    res = torch.zeros(n, K, dtype=torch.float).to(device)
    if K == 3:
        for i in range(n):
            for j in range(K):
                res[i,j] = arr[i,arr3[j]]
    elif K == 4:
        for i in range(n):
            for j in range(K):
                res[i,j] = arr[i,arr4[j]]
    elif K == 5:
        for i in range(n):
            for j in range(K):
                res[i,j] = arr[i,arr5[j]]
    elif K == 6:
        for i in range(n):
            for j in range(K):
                res[i,j] = arr[i,arr6[j]]
    elif K == 9:
        for i in range(n):
            for j in range(K):
                res[i,j] = arr[i,arr9[j]]
    elif K == 10:
        for i in range(n):
            for j in range(K):
                res[i,j] = arr[i,arr10[j]]
    return torch.FloatTensor(res)
def zigzag_rev(arr, K, device):
    n = len(arr)
    res = torch.zeros(n, dtype=torch.long).to(device)
    if K == 3:
        for i in range(n):
            res[i] = arr3.index(int(arr[i].item()))
    elif K == 4:
        for i in range(n):
            res[i] = arr4.index(int(arr[i].item()))
    elif K == 5:
        for i in range(n):
            res[i] = arr5.index(int(arr[i].item()))
    elif K == 6:
        for i in range(n):
            res[i] = arr6.index(int(arr[i].item()))
    elif K == 9:
        for i in range(n):
            res[i] = arr9.index(int(arr[i].item()))
    elif K == 10:
        for i in range(n):
            res[i] = arr10.index(int(arr[i].item()))
    return res

#====================  loader  ====================#
def train_test_loader(train_data, valid_data, test_data, K, BS, OP, device):
    train_X,  valid_X,  test_X  = torch.FloatTensor(train_data[:,:-1]), torch.FloatTensor(valid_data[:,:-1]), torch.FloatTensor(test_data[:,:-1])
    train_YO, valid_YO, test_YO = torch.LongTensor(train_data[:,-1]),   torch.LongTensor(valid_data[:,-1]),   torch.LongTensor(test_data[:,-1])
    train_BS, valid_BS, test_BS = BS, 100, 100
    if OP=='O':
        train_loader = DataLoader(TensorDataset(train_X, train_YO, train_YO), batch_size=train_BS, shuffle=True, drop_last=True)
        train_loader2= DataLoader(TensorDataset(train_X, train_YO, train_YO), batch_size=train_BS)
        valid_loader = DataLoader(TensorDataset(valid_X, valid_YO, valid_YO), batch_size=valid_BS)
        test_loader  = DataLoader(TensorDataset(test_X,  test_YO,  test_YO),  batch_size=test_BS)
    elif OP=='P':
        train_YP, valid_YP, test_YP = zigzag(train_YO, K, device), zigzag(valid_YO, K, device), zigzag(test_YO, K, device)
        train_loader = DataLoader(TensorDataset(train_X, train_YP, train_YO), batch_size=train_BS, shuffle=True, drop_last=True)
        train_loader2= DataLoader(TensorDataset(train_X, train_YP, train_YO), batch_size=train_BS)
        valid_loader = DataLoader(TensorDataset(valid_X, valid_YP, valid_YO), batch_size=valid_BS)
        test_loader  = DataLoader(TensorDataset(test_X,  test_YP,  test_YO),  batch_size=test_BS)
    return train_loader, train_loader2, valid_loader, test_loader


#====================  Unimodality  ====================#
def unimodality(pro, K):
    unimo = 0.
    amaxp = pro.argmax(dim=1)
    for i in range(len(pro)):
        flag = 1.
        for k in range(max([0, amaxp[i]-K]), amaxp[i]):
            if pro[i,k] > pro[i,k+1]:
                flag = 0.
        for k in range(amaxp[i], min([K-1, amaxp[i]+K])):
            if pro[i,k] < pro[i,k+1]:
                flag = 0.
        unimo += flag
    return unimo


#====================  labeling  ====================#
def N_all_gy(loader, device, model):
    model.eval()
    all_g = torch.tensor([], dtype=torch.float).to(device)
    all_y = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for X, Y, _ in loader:
            X, Y = X.to(device), Y.to(device)
            #
            g = model(X)
            #
            all_g = torch.cat((all_g, g))
            all_y = torch.cat((all_y, Y))
    #all_g, indeces = torch.sort(all_g,0)
    #all_y = all_y[indeces.reshape(-1)]
    return all_g, all_y
def B_all_gy(loader, device, model):
    model.eval()
    all_g = torch.tensor([], dtype=torch.float).to(device)
    all_y = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for X, Y, _ in loader:
            X, Y = X.to(device), Y.to(device)
            #
            g, _ = model(X)
            #
            all_g = torch.cat((all_g, g))
            all_y = torch.cat((all_y, Y))
    #all_g, indeces = torch.sort(all_g,0)
    #all_y = all_y[indeces.reshape(-1)]
    return all_g, all_y
def SC_N_all_gy(loader, device, model):
    model.eval()
    all_g = torch.tensor([], dtype=torch.float).to(device)
    all_y = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for X, Y, _ in loader:
            X, Y = X.to(device), Y.to(device)
            #
            _, g = model(X)
            #
            all_g = torch.cat((all_g, g))
            all_y = torch.cat((all_y, Y))
    #all_g, indeces = torch.sort(all_g,0)
    #all_y = all_y[indeces.reshape(-1)]
    return all_g, all_y
def SC_B_all_gy(loader, device, model):
    model.eval()
    all_g = torch.tensor([], dtype=torch.float).to(device)
    all_y = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for X, Y, _ in loader:
            X, Y = X.to(device), Y.to(device)
            #
            _, g, _ = model(X)
            #
            all_g = torch.cat((all_g, g))
            all_y = torch.cat((all_y, Y))
    #all_g, indeces = torch.sort(all_g,0)
    #all_y = all_y[indeces.reshape(-1)]
    return all_g, all_y
def OT_func(a, y, K, device, ZAS):
    #task loss
    loss = torch.zeros(K, K, dtype=torch.float).to(device)
    if ZAS=="Z":
        for j, k in product(range(K),range(K)):
            if j!=k: loss[j,k] = 1.
    if ZAS=="A":
        for j, k in product(range(K),range(K)):
            loss[j,k] = abs(j-k)
    if ZAS=="S":
        for j, k in product(range(K),range(K)):
            loss[j,k] = (j-k)**2
    #sort (a,y)
    a, idx = torch.sort(a.reshape(-1))
    y = y[idx]
    #all size n1, unique size n2
    n1 = a.shape[0]
    ua = a.unique(sorted=True).to(dtype=torch.float64)
    n2 = ua.shape[0]
    #DP matrix
    L  = torch.zeros(n2, K, dtype=torch.float64).to(device)
    M  = torch.zeros(n2, K, dtype=torch.long).to(device)
    #
    Ys = y[a==ua[0]]
    for k in range(K):
        L[0,k] = torch.sum(loss[Ys,k])
        M[0,k] = torch.argmin(L[0,:k+1])
    #
    for j in range(1,n2):
        Ys = y[a==ua[j]]
        for k in range(K):
            L[j,k] = torch.min(L[j-1,:k+1]) + torch.sum(loss[Ys,k])
            M[j,k] = torch.argmin(L[j,:k+1])
    #threshold parameters
    t = torch.zeros(K-1, dtype=torch.float64).to(device)
    #
    I = M[-1,-1].item()
    for k in range(I,K-1): t[k] = 10.**8
    #
    for j in reversed(range(n2-1)):
        J = M[j,I].item()
        if I!=J:
            for k in range(J,I): t[k] = (ua[j]+ua[j+1])*0.5
            I = J
    #
    for k in range(I): t[k] = -10.**8
    return t
def OT_pred(K, OP, device, g, t_Z, t_A, t_S):
    #task Z
    pred_Z = torch.sum(t_Z-g<=0., 1)
    if OP=="P": pred_Z = zigzag_rev(pred_Z, K, device)
    #task A
    pred_A = torch.sum(t_A-g<=0., 1)
    if OP=="P": pred_A = zigzag_rev(pred_A, K, device)
    #task S
    pred_S = torch.sum(t_S-g<=0., 1)
    if OP=="P": pred_S = zigzag_rev(pred_S, K, device)
    #results: 3
    return pred_Z, pred_A, pred_S
def MT_pred(K, OP, device, g, b):
    #task Z,A,S
    tmp = torch.zeros(g.shape[0],K-1).to(device)
    tmp[b-g<=0.] = 1.
    tmp = torch.cat([tmp, torch.zeros(g.shape[0],1).to(device)], 1)
    pred = torch.argmin(tmp, 1)
    if OP=="P": pred = zigzag_rev(pred, K, device)
    pred_Z = pred; pred_A = pred; pred_S = pred
    #results: 3
    return pred_Z, pred_A, pred_S
def ST_pred(K, OP, device, g, b):
    #task Z,A,S
    pred = torch.sum(b-g<=0., 1)
    if OP=="P": pred = zigzag_rev(pred, K, device)
    pred_Z = pred; pred_A = pred; pred_S = pred
    #results: 3
    return pred_Z, pred_A, pred_S
def NT_pred(K, OP, device, g):
    #task Z,A,S
    t = torch.arange(K-1)+0.5
    pred = torch.sum(t-g<=0., 1)
    if OP=="P": pred = zigzag_rev(pred, K, device)
    pred_Z = pred; pred_A = pred; pred_S = pred
    #results: 3
    return pred_Z, pred_A, pred_S
def LB_pred(K, OP, device, prob):
    if OP=="P": prob = zigzag_inv(prob, K, device)
    #task Z
    pred_Z = prob.argmax(dim=1)
    #task A
    L_A = torch.zeros(K, K, dtype=torch.float).to(device)
    for j, k in product(range(K),range(K)): L_A[j,k] = abs(j-k)
    pred_A = torch.LongTensor(torch.argmin(torch.mm(prob, L_A), dim=1))
    #task S
    L_S = torch.zeros(K, K, dtype=torch.float).to(device)
    for j, k in product(range(K),range(K)): L_S[j,k] = (j-k)**2
    pred_S = torch.LongTensor(torch.argmin(torch.mm(prob, L_S), dim=1))
    #results: 3
    return pred_Z, pred_A, pred_S


#====================  ZAS error  ====================#
def ZAS_error(pred_Z, pred_A, pred_S, Y, LL=True):
    MZE_Z = torch.sum(Y != pred_Z).float()
    MAE_A = torch.sum(torch.abs(Y.float() - pred_A.float()))
    MSE_S = torch.sum(torch.pow(Y.float() - pred_S.float(), 2))
    #results: 3
    return MZE_Z, MAE_A, MSE_S


#====================  SC (統計的 分類器)  ====================#
#モデル: {SL, OCL}
#損失: {NLL}
#ラベリング関数: {分類器の尤度ベースCL}
#============================================================#
class FD(nn.Module):
    def __init__(self, d, M, K):
        super(FD, self).__init__()
        self.gC1 = nn.Linear(d, M); #torch.nn.init.normal_(self.gC1.weight, mean=0., std=.5); #torch.nn.init.normal_(self.gC1.bias, mean=0., std=.5)
        self.gC2 = nn.Linear(M, M); #torch.nn.init.normal_(self.gC2.weight, mean=0., std=.5); #torch.nn.init.normal_(self.gC2.bias, mean=0., std=.5)
        self.gC3 = nn.Linear(M, M); #torch.nn.init.normal_(self.gC3.weight, mean=0., std=.5); #torch.nn.init.normal_(self.gC3.bias, mean=0., std=.5)
        self.gC4 = nn.Linear(M, K); #torch.nn.init.normal_(self.gC4.weight, mean=0., std=.5); #torch.nn.init.normal_(self.gC4.bias, mean=0., std=.5)
    def forward(self, x):
        gC = torch.sigmoid(self.gC1(x))
        gC = torch.sigmoid(self.gC2(gC))
        gC = torch.sigmoid(self.gC3(gC))
        gC = self.gC4(gC)
        return gC
#full-DF ordered cumulative logit model
def train_FDOCL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC1 = model(X)
        gC2 = torch.zeros(gC1.shape[0], K-1).float().to(device)
        gC2[:,0] = gC1[:,0]
        for k in range(1, K-1): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
        gC2 = torch.sigmoid(gC2)
        probC = torch.cat([gC2, torch.ones(gC2.shape[0],1).to(device)], dim=1) - torch.cat([torch.zeros(gC2.shape[0],1).to(device), gC2], dim=1)
        probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(probC), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_FDOCL_NLL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC1 = model(X)
            gC2 = torch.zeros(gC1.shape[0], K-1).float().to(device)
            gC2[:,0] = gC1[:,0]
            for k in range(1, K-1): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
            gC2 = torch.sigmoid(gC2)
            probC = torch.cat([gC2, torch.ones(gC2.shape[0],1).to(device)], dim=1) - torch.cat([torch.zeros(gC2.shape[0],1).to(device), gC2], dim=1)
            probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(probC), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF adjacent categories logit model
def train_FDACL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC1 = model(X)
        gC2 = torch.zeros(gC1.shape[0], K).float().to(device)
        for k in range(1,K): gC2[:,k] = gC2[:,k-1] + gC1[:,k-1]
        probC = (-gC2).softmax(dim=1)
        probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(probC), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_FDACL_NLL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC1 = model(X)
            gC2 = torch.zeros(gC1.shape[0], K).float().to(device)
            for k in range(1,K): gC2[:,k] = gC2[:,k-1] + gC1[:,k-1]
            probC = (-gC2).softmax(dim=1)
            probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(probC), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF ordered adjacent categories logit model
def train_FDOACL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC1 = model(X)
        gC2 = torch.zeros(gC1.shape[0], K-1).float().to(device)
        gC2[:,0] = gC1[:,0]
        for k in range(1,K-1): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
        gC3 = torch.zeros(gC1.shape[0], K).float().to(device)
        for k in range(1,K): gC3[:,k] = gC3[:,k-1] + gC2[:,k-1]
        probC = (-gC3).softmax(dim=1)
        probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(probC), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_FDOACL_NLL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC1 = model(X)
            gC2 = torch.zeros(gC1.shape[0], K-1).float().to(device)
            gC2[:,0] = gC1[:,0]
            for k in range(1,K-1): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
            gC3 = torch.zeros(gC1.shape[0], K).float().to(device)
            for k in range(1,K): gC3[:,k] = gC3[:,k-1] + gC2[:,k-1]
            probC = (-gC3).softmax(dim=1)
            probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(probC), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF continuous ratio logit model
def train_FDCRL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC1 = model(X)
        gC2 = torch.cat([gC1, 10.**8*torch.ones(gC1.shape[0],1).float().to(device)], dim=1)
        prob = torch.ones(gC1.shape[0], K).float().to(device)
        for k in range(K):
            for j in range(k): prob[:,k] *= torch.sigmoid(-gC2[:,j])
            prob[:,k] *= torch.sigmoid(gC2[:,k])
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(probC), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_FDCRL_NLL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC1 = model(X)
            gC2 = torch.cat([gC1, 10.**8*torch.ones(gC1.shape[0],1).float().to(device)], dim=1)
            prob = torch.ones(gC1.shape[0], K).float().to(device)
            for k in range(K):
                for j in range(k): prob[:,k] *= torch.sigmoid(-gC2[:,j])
                prob[:,k] *= torch.sigmoid(gC2[:,k])
            prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(probC), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF ordered continuous ratio logit model
def train_FDOCRL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC1 = model(X)
        gC2 = torch.zeros(gC1.shape[0], K).float().to(device)
        gC2[:,0] = gC1[:,0]
        for k in range(1,K): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
        gC3 = torch.cat([gC2, 10.**8*torch.ones(gC2.shape[0],1).float().to(device)], dim=1)
        prob = torch.ones(gC3.shape[0], K).float().to(device)
        for k in range(K):
            for j in range(k): prob[:,k] *= torch.sigmoid(-gC3[:,j])
            prob[:,k] *= torch.sigmoid(gC3[:,k])
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(probC), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_FDOCRL_NLL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC1 = model(X)
            gC2 = torch.zeros(gC1.shape[0], K).float().to(device)
            gC2[:,0] = gC1[:,0]
            for k in range(1,K): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
            gC3 = torch.cat([gC2, 10.**8*torch.ones(gC2.shape[0],1).float().to(device)], dim=1)
            prob = torch.ones(gC3.shape[0], K).float().to(device)
            for k in range(K):
                for j in range(k): prob[:,k] *= torch.sigmoid(-gC3[:,j])
                prob[:,k] *= torch.sigmoid(gC3[:,k])
            prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(probC), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF stereotype logit model
def train_FDSL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC = model(X)
        probC = gC.softmax(dim=1)
        probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(probC), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_FDSL_NLL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC = model(X)
            probC = gC.softmax(dim=1)
            probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(probC), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF V-shaped stereotype logit model
def train_FDVSL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC1 = model(X)
        gC2 = torch.zeros(gC1.shape[0], K).float().to(device)
        gC2[:,0] = gC1[:,0]
        for k in range(1, K): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
        probC = (-torch.square(gC2)).softmax(dim=1)
        probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(probC), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_FDVSL_NLL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC1 = model(X)
            gC2 = torch.zeros(gC1.shape[0], K).float().to(device)
            gC2[:,0] = gC1[:,0]
            for k in range(1, K): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
            probC = (-torch.square(gC2)).softmax(dim=1)
            probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(probC), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF cumulative logit model
def train_FDCL_ANLCL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC1 = model(X)
        gC2 = torch.zeros(gC1.shape[0], K-1).float().to(device)
        gC2[:,0] = gC1[:,0]
        for k in range(1, K-1): gC2[:,k] = gC2[:,k-1] + gC1[:,k]
        gC2 = torch.sigmoid(gC2)
        probC = torch.cat([gC2, torch.ones(gC2.shape[0],1).to(device)], dim=1) - torch.cat([torch.zeros(gC2.shape[0],1).to(device), gC2], dim=1)
        #probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        tmp = probC[:,0].reshape(-1,1)
        tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(tmp), (Y>0).type(torch.uint8).long() )
        for k in range(1,K-1):
            tmp = torch.sum(probC[:,:k+1], dim=1).reshape(-1,1)
            tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
            loss += F.nll_loss(torch.log(tmp), (Y>k).type(torch.uint8).long() )
        #learning
        loss.backward()
        optimizer.step()
def test_FDCL_ANLCL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC1 = model(X)
            gC2 = torch.zeros(gC1.shape[0], K-1).float().to(device)
            gC2[:,0] = gC1[:,0]
            for k in range(1, K-1): gC2[:,k] = gC2[:,k-1] + gC1[:,k]
            gC2 = torch.sigmoid(gC2)
            probC = torch.cat([gC2, torch.ones(gC2.shape[0],1).to(device)], dim=1) - torch.cat([torch.zeros(gC2.shape[0],1).to(device), gC2], dim=1)
            #probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                for k in range(K-1):
                    tmp = torch.sum(probC[:,:k+1], dim=1).reshape(-1,1)
                    tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
                    loss += F.nll_loss(torch.log(tmp), (Y1>k).type(torch.uint8).long() , reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF ordered cumulative logit model
def train_FDOCL_ANLCL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC1 = model(X)
        gC2 = torch.zeros(gC1.shape[0], K-1).float().to(device)
        gC2[:,0] = gC1[:,0]
        for k in range(1, K-1): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
        gC2 = torch.sigmoid(gC2)
        probC = torch.cat([gC2, torch.ones(gC2.shape[0],1).to(device)], dim=1) - torch.cat([torch.zeros(gC2.shape[0],1).to(device), gC2], dim=1)
        #probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        tmp = probC[:,0].reshape(-1,1)
        tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(tmp), (Y>0).type(torch.uint8).long() )
        for k in range(1,K-1):
            tmp = torch.sum(probC[:,:k+1], dim=1).reshape(-1,1)
            tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
            loss += F.nll_loss(torch.log(tmp), (Y>k).type(torch.uint8).long() )
        #learning
        loss.backward()
        optimizer.step()
def test_FDOCL_ANLCL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC1 = model(X)
            gC2 = torch.zeros(gC1.shape[0], K-1).float().to(device)
            gC2[:,0] = gC1[:,0]
            for k in range(1, K-1): gC2[:,k] = gC2[:,k-1] + torch.pow(gC1[:,k],2)
            gC2 = torch.sigmoid(gC2)
            probC = torch.cat([gC2, torch.ones(gC2.shape[0],1).to(device)], dim=1) - torch.cat([torch.zeros(gC2.shape[0],1).to(device), gC2], dim=1)
            #probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                for k in range(K-1):
                    tmp = torch.sum(probC[:,:k+1], dim=1).reshape(-1,1)
                    tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
                    loss += F.nll_loss(torch.log(tmp), (Y1>k).type(torch.uint8).long() , reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#full-DF stereotype logit model
def train_FDSL_ANLCL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC = model(X)
        probC = gC.softmax(dim=1)
        #probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        tmp = probC[:,0].reshape(-1,1)
        tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(tmp), (Y>0).type(torch.uint8).long() )
        for k in range(1,K-1):
            tmp = torch.sum(probC[:,:k+1], dim=1).reshape(-1,1)
            tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
            loss += F.nll_loss(torch.log(tmp), (Y>k).type(torch.uint8).long() )
        #learning
        loss.backward()
        optimizer.step()
def test_FDSL_ANLCL(loader, K, OP, device, model, LAB, LL=True):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC = model(X)
            probC = gC.softmax(dim=1)
            #probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                for k in range(K-1):
                    tmp = torch.sum(probC[:,:k+1], dim=1).reshape(-1,1)
                    tmp = torch.clamp(torch.cat([tmp,1.-tmp], dim=1), min=.1**30, max=1.-.1**30)
                    loss += F.nll_loss(torch.log(tmp), (Y1>k).type(torch.uint8).long() , reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n


#====================  SC (統計的 分類器) + NN (非バイアスパラメトリック 非統計的 回帰器)  ====================#
#{SL}モデル + {NN}モデル
#{NLL}損失
#ラベリング関数(LAB)は, 統計的 分類器の{SC}, 非バイアスパラメトリック 非統計的 回帰器の{IT, NT}
#============================================================#
#stereotype logit (SL) model + absolute-deviation (AD) model
class FD_ODN(nn.Module):
    def __init__(self, d, M, K):
        super(FD_ODN, self).__init__()
        self.g1S = nn.Linear(d, M); #torch.nn.init.normal_(self.g1S.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g1S.bias, mean=0., std=.5)
        self.g2S = nn.Linear(M, M); #torch.nn.init.normal_(self.g2S.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g2S.bias, mean=0., std=.5)
        self.g3C = nn.Linear(M, M); #torch.nn.init.normal_(self.g3C.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g3C.bias, mean=0., std=.5)
        self.g4C = nn.Linear(M, K); #torch.nn.init.normal_(self.g4C.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g4C.bias, mean=0., std=.5)
        self.g3R = nn.Linear(M, M); #torch.nn.init.normal_(self.g3R.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g3R.bias, mean=0., std=.5)
        self.g4R = nn.Linear(M, 1); #torch.nn.init.normal_(self.g4R.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g4R.bias, mean=0., std=.5)
        self.K = K
    def forward(self, x):
        gS = torch.sigmoid(self.g1S(x))
        gS = torch.sigmoid(self.g2S(gS))
        gC = torch.sigmoid(self.g3C(gS))
        gC = self.g4C(gC)
        gR = torch.sigmoid(self.g3R(gS))
        gR = self.g4R(gR) + (self.K-1.)/2.
        return gC, gR
def train_FDSL_NLL_ODN_AD(loader, K, device, model, rlam, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        gC, gR = model(X)
        probC = gC.softmax(dim=1)
        probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
        lossC = F.nll_loss(torch.log(probC), Y)
        lossR = F.l1_loss(gR, Y.reshape(-1,1).float())
        loss = lossC + rlam*lossR
        #learning
        loss.backward()
        optimizer.step()
def test_FDSL_NLL_ODN_AD(loader, K, OP, device, model, rlam, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, lossC, lossR, loss = 0, 0., 0., 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            gC, gR = model(X)
            probC = gC.softmax(dim=1)
            probC = torch.clamp(probC, min=.1**30, max=1.-.1**30)
            if LL==True:
                lossC += F.nll_loss(torch.log(probC), Y1, reduction='sum')
                lossR += F.l1_loss(gR, Y1.reshape(-1,1).float(), reduction='sum')
                loss = lossC+rlam*lossR
            #label prediction
            if LL==False:
                if LAB=="CL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, probC)
                if LAB=='NT': pred_Z, pred_A, pred_S = NT_pred(K, OP, device, gR)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, gR, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 3, 3, 9
    if LL==True:  return lossC/n, lossR/n, loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n


#====================  回帰モデル  ====================#
class ODN(nn.Module):
    def __init__(self, d, M, K):
        super(ODN, self).__init__()
        self.gR1 = nn.Linear(d, M); #torch.nn.init.normal_(self.gR1.weight, mean=0., std=.5); #torch.nn.init.normal_(self.gR1.bias, mean=0., std=.5)
        self.gR2 = nn.Linear(M, M); #torch.nn.init.normal_(self.gR2.weight, mean=0., std=.5); #torch.nn.init.normal_(self.gR2.bias, mean=0., std=.5)
        self.gR3 = nn.Linear(M, M); #torch.nn.init.normal_(self.gR3.weight, mean=0., std=.5); #torch.nn.init.normal_(self.gR3.bias, mean=0., std=.5)
        self.gR4 = nn.Linear(M, 1); #torch.nn.init.normal_(self.gR4.weight, mean=0., std=.5); #torch.nn.init.normal_(self.gR4.bias, mean=0., std=.5)
        self.K = K
    def forward(self, x):
        gR = torch.sigmoid(self.gR1(x))
        gR = torch.sigmoid(self.gR2(gR))
        gR = torch.sigmoid(self.gR3(gR))
        gR = self.gR4(gR) + self.K/2.
        return gR
class ODB(nn.Module):
    def __init__(self, d, M, K):
        super(ODB, self).__init__()
        self.g1 = nn.Linear(d, M); #torch.nn.init.normal_(self.g1.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g1.bias, mean=0., std=.5)
        self.g2 = nn.Linear(M, M); #torch.nn.init.normal_(self.g2.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g2.bias, mean=0., std=.5)
        self.g3 = nn.Linear(M, M); #torch.nn.init.normal_(self.g3.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g3.bias, mean=0., std=.5)
        self.g4 = nn.Linear(M, 1); #torch.nn.init.normal_(self.g4.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g4.bias, mean=0., std=.5)
        self.bi  = nn.Parameter(torch.arange(1,K).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g) + self.K/2.
        b = torch.cat([torch.tensor([0.]), self.bi])
        return g, b
class ODOB(nn.Module):
    def __init__(self, d, M, K):
        super(ODOB, self).__init__()
        self.g1 = nn.Linear(d, M); #torch.nn.init.normal_(self.g1.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g1.bias, mean=0., std=.5)
        self.g2 = nn.Linear(M, M); #torch.nn.init.normal_(self.g2.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g2.bias, mean=0., std=.5)
        self.g3 = nn.Linear(M, M); #torch.nn.init.normal_(self.g3.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g3.bias, mean=0., std=.5)
        self.g4 = nn.Linear(M, 1); #torch.nn.init.normal_(self.g4.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g4.bias, mean=0., std=.5)
        self.bi  = nn.Parameter(torch.ones(K-1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g) + self.K/2.
        b = torch.zeros(self.K).float()
        for k in range(1, self.K): b[k] = b[k-1] + torch.pow(self.bi[k-1],2)
        return g, b
class ODS(nn.Module):
    def __init__(self, d, M, K):
        super(ODS, self).__init__()
        self.g1 = nn.Linear(d, M); #torch.nn.init.normal_(self.g1.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g1.bias, mean=0., std=.5)
        self.g2 = nn.Linear(M, M); #torch.nn.init.normal_(self.g2.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g2.bias, mean=0., std=.5)
        self.g3 = nn.Linear(M, M); #torch.nn.init.normal_(self.g3.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g3.bias, mean=0., std=.5)
        self.g4 = nn.Linear(M, 1); #torch.nn.init.normal_(self.g4.weight, mean=0., std=.5); #torch.nn.init.normal_(self.g4.bias, mean=0., std=.5)
        self.sc = nn.Parameter(torch.zeros(1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g) + self.K/2.
        s = 0.01+torch.exp(self.sc)
        return g, s


#====================  NNR (非バイアスパラメトリック 非統計的 回帰器)  ====================#
#absolute-deviation (AD) model
def train_AD(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g = model(X)
        loss = F.l1_loss(g, Y.reshape(-1,1).float())
        #learning
        loss.backward()
        optimizer.step()
def test_AD(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g = model(X)
            if LL==True:
                loss += F.l1_loss(g, Y1.reshape(-1,1).float(), reduction='sum')
            #label prediction
            if LL==False:
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                if LAB=='NT': pred_Z, pred_A, pred_S = NT_pred(K, OP, device, g)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#squared (SQ) model
def train_SQ(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g = model(X)
        loss = F.mse_loss(g, Y.reshape(-1,1).float())
        #learning
        loss.backward()
        optimizer.step()
def test_SQ(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g = model(X)
            if LL==True:
                loss += F.mse_loss(g, Y1.reshape(-1,1).float(), reduction='sum')
            #label prediction
            if LL==False:
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                if LAB=='NT': pred_Z, pred_A, pred_S = NT_pred(K, OP, device, g)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n


#====================  BNR (バイアスパラメトリック 非統計的 回帰器)  ====================#
#POCL-AT
def train_LogiAT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        lev = torch.zeros(g.shape[0],K-1)
        for i in range(g.shape[0]): lev[i,:Y[i]] = 1.
        loss = -torch.mean(torch.sum((F.logsigmoid(-b+g)*lev + F.logsigmoid(b-g)*(1-lev)), 1))
        #learning
        loss.backward()
        optimizer.step()
def test_LogiAT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                lev = torch.zeros(g.shape[0], K-1).to(device)
                for i in range(g.shape[0]): lev[i,:Y1[i]] = 1.
                loss += -torch.sum(torch.sum((F.logsigmoid(-b+g)*lev + F.logsigmoid(b-g)*(1-lev)), 1))
            #label prediction
            if LL==False:
                if LAB=="RL":
                    prob = torch.sigmoid(b-g)
                    prob = torch.cat([prob, torch.ones(prob.shape[0],1).float().to(device)], dim=1) - torch.cat([torch.zeros(prob.shape[0],1).float().to(device), prob], dim=1)
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#POCL-IT
def train_LogiIT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),F.logsigmoid(-b+g)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        tmp2 = torch.cat([F.logsigmoid(b-g),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        loss = -torch.mean(tmp1+tmp2)
        #learning
        loss.backward()
        optimizer.step()
def test_LogiIT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),F.logsigmoid(-b+g)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                tmp2 = torch.cat([F.logsigmoid(b-g),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                loss += -torch.sum(tmp1+tmp2)
            #label prediction
            if LL==False:
                if LAB=="RL":
                    tmp1 = b-g
                    tmp2 = torch.zeros(tmp1.shape[0], K).float().to(device)
                    for k in range(1,K): tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
                    prob = (-tmp2).softmax(dim=1)
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#SVOR-AT
def train_HingAT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        lev = torch.zeros(g.shape[0],K-1)
        for i in range(g.shape[0]): lev[i,:Y[i]] = 1.
        loss = torch.mean(torch.sum((torch.relu(1.+b-g)*lev + torch.relu(1.-b+g)*(1-lev)), 1))
        #learning
        loss.backward()
        optimizer.step()
def test_HingAT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                lev = torch.zeros(g.shape[0], K-1).to(device)
                for i in range(g.shape[0]): lev[i,:Y1[i]] = 1.
                loss += torch.sum(torch.sum((torch.relu(1.+b-g)*lev + torch.relu(1.-b+g)*(1-lev)), 1))
            #label prediction
            if LL==False:
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#SVOR-IT
def train_HingIT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),F.relu(1.+b-g)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        tmp2 = torch.cat([F.relu(1.-b+g),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        loss = torch.mean(tmp1+tmp2)
        #learning
        loss.backward()
        optimizer.step()
def test_HingIT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),F.relu(1.+b-g)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                tmp2 = torch.cat([F.relu(1.-b+g),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                loss = torch.sum(tmp1+tmp2)
            #label prediction
            if LL==False:
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#SmHinge-AT
def train_SmHiAT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        lev = torch.zeros(g.shape[0],K-1)
        for i in range(g.shape[0]): lev[i,:Y[i]] = 1.
        arr1 = g-b; tmp1 = torch.pow(torch.relu(1.-arr1), 2); tmp1[arr1<0] = torch.relu(1.-2.*arr1)[arr1<0]
        arr2 = b-g; tmp2 = torch.pow(torch.relu(1.-arr2), 2); tmp2[arr2<0] = torch.relu(1.-2.*arr2)[arr2<0]
        loss = torch.mean(torch.sum((tmp1*lev + tmp2*(1-lev)), 1))
        #learning
        loss.backward()
        optimizer.step()
def test_SmHiAT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                lev = torch.zeros(g.shape[0], K-1).to(device)
                for i in range(g.shape[0]): lev[i,:Y1[i]] = 1.
                arr1 = g-b; tmp1 = torch.pow(torch.relu(1.-arr1), 2); tmp1[arr1<0] = torch.relu(1.-2.*arr1)[arr1<0]
                arr2 = b-g; tmp2 = torch.pow(torch.relu(1.-arr2), 2); tmp2[arr2<0] = torch.relu(1.-2.*arr2)[arr2<0]
                loss += torch.sum(torch.sum((tmp1*lev + tmp2*(1-lev)), 1))
            #label prediction
            if LL==False:
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#SmHinge-IT
def train_SmHiIT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        arr1 = g-b; tmp1 = torch.pow(torch.relu(1.-arr1), 2); tmp1[arr1<0] = torch.relu(1.-2.*arr1)[arr1<0]
        arr2 = b-g; tmp2 = torch.pow(torch.relu(1.-arr2), 2); tmp2[arr2<0] = torch.relu(1.-2.*arr2)[arr2<0]
        tmp3 = torch.cat([torch.zeros(g.shape[0],1).to(device),tmp1],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        tmp4 = torch.cat([tmp2,torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        loss = torch.mean(tmp3+tmp4)
        #learning
        loss.backward()
        optimizer.step()
def test_SmHiIT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                arr1 = g-b; tmp1 = torch.pow(torch.relu(1.-arr1), 2); tmp1[arr1<0] = torch.relu(1.-2.*arr1)[arr1<0]
                arr2 = b-g; tmp2 = torch.pow(torch.relu(1.-arr2), 2); tmp2[arr2<0] = torch.relu(1.-2.*arr2)[arr2<0]
                tmp3 = torch.cat([torch.zeros(g.shape[0],1).to(device),tmp1],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                tmp4 = torch.cat([tmp2,torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                loss += torch.sum(tmp3+tmp4)
            #label prediction
            if LL==False:
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#SmHinge-AT
def train_SqHiAT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        lev = torch.zeros(g.shape[0],K-1)
        for i in range(g.shape[0]): lev[i,:Y[i]] = 1.
        loss = torch.mean(torch.sum((torch.pow(torch.relu(1.+b-g),2)*lev + torch.pow(torch.relu(1.-b+g),2)*(1-lev)), 1))
        #learning
        loss.backward()
        optimizer.step()
def test_SqHiAT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                lev = torch.zeros(g.shape[0], K-1).to(device)
                for i in range(g.shape[0]): lev[i,:Y1[i]] = 1.
                loss += torch.sum(torch.sum((torch.pow(torch.relu(1.+b-g),2)*lev + torch.pow(torch.relu(1.-b+g),2)*(1-lev)), 1))
            #label prediction
            if LL==False:
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#SqHinge-IT
def train_SqHiIT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),torch.pow(F.relu(1.+b-g),2)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        tmp2 = torch.cat([torch.pow(F.relu(1.-b+g),2),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        loss = torch.mean(tmp1+tmp2)
        #learning
        loss.backward()
        optimizer.step()
def test_SqHiIT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),torch.pow(F.relu(1.+b-g),2)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                tmp2 = torch.cat([torch.pow(F.relu(1.-b+g),2),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                loss += torch.sum(tmp1+tmp2)
            #label prediction
            if LL==False:
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#Squared-AT
def train_SquaAT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        lev = torch.zeros(g.shape[0],K-1)
        for i in range(g.shape[0]): lev[i,:Y[i]] = 1.
        loss = torch.mean(torch.sum((torch.pow((1.+b-g),2)*lev + torch.pow((1.-b+g),2)*(1-lev)), 1))
        #learning
        loss.backward()
        optimizer.step()
def test_SquaAT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                lev = torch.zeros(g.shape[0], K-1).to(device)
                for i in range(g.shape[0]): lev[i,:Y1[i]] = 1.
                loss += torch.sum(torch.sum((torch.pow((1.+b-g),2)*lev + torch.pow((1.-b+g),2)*(1-lev)), 1))
            #label prediction
            if LL==False:
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#Squared-IT
def train_SquaIT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),torch.pow((1.+b-g),2)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        tmp2 = torch.cat([torch.pow((1.-b+g),2),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        loss = torch.mean(tmp1+tmp2)
        #learning
        loss.backward()
        optimizer.step()
def test_SquaIT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),torch.pow((1.+b-g),2)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                tmp2 = torch.cat([torch.pow((1.-b+g),2),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                loss += torch.sum(tmp1+tmp2)
            #label prediction
            if LL==False:
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#ORBoost-AT
def train_ExpoAT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        lev = torch.zeros(g.shape[0],K-1)
        for i in range(g.shape[0]): lev[i,:Y[i]] = 1.
        loss = torch.mean(torch.sum((torch.exp(b-g)*lev + torch.exp(-b+g)*(1-lev)), 1))
        #learning
        loss.backward()
        optimizer.step()
def test_ExpoAT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                lev = torch.zeros(g.shape[0], K-1).to(device)
                for i in range(g.shape[0]): lev[i,:Y1[i]] = 1.
                loss += torch.sum(torch.sum((torch.exp(b-g)*lev + torch.exp(-b+g)*(1-lev)), 1))
            #label prediction
            if LL==False:
                if LAB=="RL":
                    prob = torch.sigmoid((b-g)/2)
                    prob = torch.cat([prob, torch.ones(prob.shape[0],1).float().to(device)], dim=1) - torch.cat([torch.zeros(prob.shape[0],1).float().to(device), prob], dim=1)
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#ORBoost-IT
def train_ExpoIT(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),torch.exp(b-g)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        tmp2 = torch.cat([torch.exp(-b+g),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y.unsqueeze(1)).squeeze(1)
        loss = torch.mean(tmp1+tmp2)
        #learning
        loss.backward()
        optimizer.step()
def test_ExpoIT(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = torch.cat([torch.zeros(g.shape[0],1).to(device),torch.exp(b-g)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                tmp2 = torch.cat([torch.exp(-b+g),torch.zeros(g.shape[0],1).to(device)],dim=1).gather(dim=-1, index=Y1.unsqueeze(1)).squeeze(1)
                loss += torch.sum(tmp1+tmp2)
            #label prediction
            if LL==False:
                if LAB=="RL":
                    tmp1 = (b-g)/2.
                    tmp2 = torch.zeros(tmp1.shape[0], K).float().to(device)
                    for k in range(1,K): tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
                    prob = (-tmp2).softmax(dim=1)
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n


#====================  NSR (非バイアスパラメトリック 統計的 回帰器)  ====================#
#BIN-NLL: binomial model; NLL loss
def train_BIN_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, s = model(X)
        prob = torch.zeros(g.shape[0], K).float().to(device)
        for k in range(K): prob[:,k] = (np.log(comb(int(K-1), k, exact=True)) + k*torch.sigmoid(g[:,0]) + (K-1-k)*torch.sigmoid(-g[:,0]))/s
        prob = prob.softmax(dim=1)
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(prob), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_BIN_NLL(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, s = model(X)
            prob = torch.zeros(g.shape[0], K).float().to(device)
            for k in range(K): prob[:,k] = (np.log(comb(int(K-1), k, exact=True)) + k*torch.sigmoid(g[:,0]) + (K-1-k)*torch.sigmoid(-g[:,0]))/s
            prob = prob.softmax(dim=1)
            prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(prob), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="RL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#POI-NLL: poisson model; NLL loss
def train_POI_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, s = model(X)
        prob = torch.zeros(g.shape[0], K).float().to(device)
        tmp = 1.
        for k in range(K):
            if k!=0: tmp += np.log(k)
            prob[:,k] = tmp-k*g[:,0]
        prob = -prob/s
        prob = prob.softmax(dim=1)
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(prob), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_POI_NLL(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, s = model(X)
            prob = torch.zeros(g.shape[0], K).float().to(device)
            tmp = 1.
            for k in range(K):
                if k!=0: tmp += np.log(k)
                prob[:,k] = tmp-k*g[:,0]
            prob = -prob/s
            prob = prob.softmax(dim=1)
            prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
            if LL==True:
                loss += F.nll_loss(torch.log(prob), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="RL": pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n


#====================  BSR (バイアスパラメトリック 統計的 回帰器)  ====================#
#PO-OCL-NLL: proportional-odds ordered cumulative logit model; NLL loss
def train_POOCL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        prob = torch.sigmoid(b-g)
        prob = torch.cat([prob, torch.ones(prob.shape[0],1).float().to(device)], dim=1) - torch.cat([torch.zeros(prob.shape[0],1).float().to(device), prob], dim=1)
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(prob), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_POOCL_NLL(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                prob = torch.sigmoid(b-g)
                prob = torch.cat([prob, torch.ones(prob.shape[0],1).float().to(device)], dim=1) - torch.cat([torch.zeros(prob.shape[0],1).float().to(device), prob], dim=1)
                prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                loss += F.nll_loss(torch.log(prob), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="RL":
                    prob = torch.sigmoid(b-g)
                    prob = torch.cat([prob, torch.ones(prob.shape[0],1).float().to(device)], dim=1) - torch.cat([torch.zeros(prob.shape[0],1).float().to(device), prob], dim=1)
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#PO-ACL-NLL: proportional-odds adjacent categories logit model; NLL loss
def train_POACL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = b-g
        tmp2 = torch.zeros(tmp1.shape[0], K).float().to(device)
        for k in range(1,K): tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
        prob = (-tmp2).softmax(dim=1)
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(prob), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_POACL_NLL(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = b-g
                tmp2 = torch.zeros(tmp1.shape[0], K).float().to(device)
                for k in range(1,K): tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
                prob = (-tmp2).softmax(dim=1)
                prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                loss += F.nll_loss(torch.log(prob), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="RL":
                    tmp1 = b-g
                    tmp2 = torch.zeros(tmp1.shape[0], K).float().to(device)
                    for k in range(1,K): tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
                    prob = (-tmp2).softmax(dim=1)
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#PO-OACL-NLL: proportional-odds ordered adjacent categories logit model; NLL loss
def train_POOACL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = b-g
        tmp2 = torch.zeros(tmp1.shape[0], K).float().to(device)
        for k in range(1,K): tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
        prob = (-tmp2).softmax(dim=1)
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(prob), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_POOACL_NLL(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = b-g
                tmp2 = torch.zeros(tmp1.shape[0], K).float().to(device)
                for k in range(1,K): tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
                prob = (-tmp2).softmax(dim=1)
                prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                loss += F.nll_loss(torch.log(prob), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="RL":
                    tmp1 = b-g
                    tmp2 = torch.zeros(tmp1.shape[0], K).float().to(device)
                    for k in range(1,K): tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
                    prob = (-tmp2).softmax(dim=1)
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#PO-CRL-NLL: proportional-odds continuous ratio logit model; NLL loss
def train_POCRL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = b-g
        tmp2 = torch.cat([tmp1, 10.**8*torch.ones(tmp1.shape[0],1).float().to(device)], dim=1)
        prob = torch.ones(tmp1.shape[0], K).float().to(device)
        for k in range(K):
            for j in range(k): prob[:,k] *= torch.sigmoid(-tmp2[:,j])
            prob[:,k] *= torch.sigmoid(tmp2[:,k])
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(prob), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_POCRL_NLL(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = b-g
                tmp2 = torch.cat([tmp1, 10.**8*torch.ones(tmp1.shape[0],1).float().to(device)], dim=1)
                prob = torch.ones(tmp1.shape[0], K).float().to(device)
                for k in range(K):
                    for j in range(k): prob[:,k] *= torch.sigmoid(-tmp2[:,j])
                    prob[:,k] *= torch.sigmoid(tmp2[:,k])
                prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                loss += F.nll_loss(torch.log(prob), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="RL":
                    tmp1 = b-g
                    tmp2 = torch.cat([tmp1, 10.**8*torch.ones(tmp1.shape[0],1).float().to(device)], dim=1)
                    prob = torch.ones(tmp1.shape[0], K).float().to(device)
                    for k in range(K):
                        for j in range(k): prob[:,k] *= torch.sigmoid(-tmp2[:,j])
                        prob[:,k] *= torch.sigmoid(tmp2[:,k])
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#PO-OCRL-NLL: proportional-odds continuous ratio logit model; NLL loss
def train_POOCRL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        tmp1 = b-g
        tmp2 = torch.cat([tmp1, 10.**8*torch.ones(tmp1.shape[0],1).float().to(device)], dim=1)
        prob = torch.ones(tmp1.shape[0], K).float().to(device)
        for k in range(K):
            for j in range(k): prob[:,k] *= torch.sigmoid(-tmp2[:,j])
            prob[:,k] *= torch.sigmoid(tmp2[:,k])
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(prob), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_POOCRL_NLL(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                tmp1 = b-g
                tmp2 = torch.cat([tmp1, 10.**8*torch.ones(tmp1.shape[0],1).float().to(device)], dim=1)
                prob = torch.ones(tmp1.shape[0], K).float().to(device)
                for k in range(K):
                    for j in range(k): prob[:,k] *= torch.sigmoid(-tmp2[:,j])
                    prob[:,k] *= torch.sigmoid(tmp2[:,k])
                prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                loss += F.nll_loss(torch.log(prob), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="RL":
                    tmp1 = b-g
                    tmp2 = torch.cat([tmp1, 10.**8*torch.ones(tmp1.shape[0],1).float().to(device)], dim=1)
                    prob = torch.ones(tmp1.shape[0], K).float().to(device)
                    for k in range(K):
                        for j in range(k): prob[:,k] *= torch.sigmoid(-tmp2[:,j])
                        prob[:,k] *= torch.sigmoid(tmp2[:,k])
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='MT': pred_Z, pred_A, pred_S = MT_pred(K, OP, device, g, b)
                if LAB=='ST': pred_Z, pred_A, pred_S = ST_pred(K, OP, device, g, b)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n
#PO-VSL-NLL: proportional-odds v-shaped stereotype logit model; NLL loss
def train_POVSL_NLL(loader, K, device, model, optimizer):
    model.train()
    for batch_idx, (X, Y, _) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        #learning loss
        g, b = model(X)
        prob = -torch.square(b-g)
        prob = prob.softmax(dim=1)
        prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(prob), Y)
        #learning
        loss.backward()
        optimizer.step()
def test_POVSL_NLL(loader, K, OP, device, model, LAB, LL=True, t_Z=None, t_A=None, t_S=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_A, MSE_S = 0., 0., 0.
    #
    with torch.no_grad():
        for X, Y1, Y2 in loader:
            X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
            n += X.shape[0]
            #learning loss
            g, b = model(X)
            if LL==True:
                prob = -torch.square(b-g)
                prob = prob.softmax(dim=1)
                prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                loss += F.nll_loss(torch.log(prob), Y1, reduction='sum')
            #label prediction
            if LL==False:
                if LAB=="RL":
                    prob = -torch.square(b-g)
                    prob = prob.softmax(dim=1)
                    prob = torch.clamp(prob, min=.1**30, max=1.-.1**30)
                    pred_Z, pred_A, pred_S = LB_pred(K, OP, device, prob)
                if LAB=='OT': pred_Z, pred_A, pred_S = OT_pred(K, OP, device, g, t_Z, t_A, t_S)
                #labeling error
                ZZ, AA, SS = ZAS_error(pred_Z, pred_A, pred_S, Y2)
                MZE_Z+=ZZ; MAE_A+=AA; MSE_S+=SS
    #out: 1, 3
    if LL==True:  return loss/n
    if LL==False: return MZE_Z/n, MAE_A/n, MSE_S/n