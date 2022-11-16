##############################
# coding: utf-8
# use like > nohup python afad-ord-logi-IT.py --cuda 0 &
##############################
# Imports
##############################
import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
from itertools import product
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
torch.backends.cudnn.deterministic = True

CSV_PATH = './afad_all.csv'
IMAGE_PATH = '../datasets/tarball-master/AFAD-Full'

for RANDOM_SEED in range(10):
    ##############################
    # Args
    ##############################
    # Argparse helper
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--numworkers', type=int, default=16)
    args = parser.parse_args()
    #
    if args.cuda >= 0: DEVICE = torch.device("cuda:%d" % args.cuda)
    else: DEVICE = torch.device("cpu")
    #
    NUM_WORKERS = args.numworkers
    #
    PATH = "result/ord-logi-IT/seed"+str(RANDOM_SEED)
    if not os.path.exists(PATH): os.makedirs(PATH)
    #
    LOGFILE_LO = os.path.join(PATH, 'training_LO.log')
    with open(LOGFILE_LO, 'w') as f: pass
    LOGFILE_RL = os.path.join(PATH, 'training_RL.log')
    with open(LOGFILE_RL, 'w') as f: pass
    LOGFILE_ST = os.path.join(PATH, 'training_ST.log')
    with open(LOGFILE_ST, 'w') as f: pass
    LOGFILE_OT = os.path.join(PATH, 'training_OT.log')
    with open(LOGFILE_OT, 'w') as f: pass
    #
    LOGFILE_B = os.path.join(PATH, 'bias.log')
    with open(LOGFILE_B, 'w') as f: pass
    LOGFILE_Z = os.path.join(PATH, 'threshold_Z.log')
    with open(LOGFILE_Z, 'w') as f: pass
    LOGFILE_A = os.path.join(PATH, 'threshold_A.log')
    with open(LOGFILE_A, 'w') as f: pass
    LOGFILE_S = os.path.join(PATH, 'threshold_S.log')
    with open(LOGFILE_S, 'w') as f: pass


    ##############################
    # Settings
    ##############################
    # Hyperparameters
    learning_rate = .1**2.5
    NUM_EPOCHS = 100

    # Architecture
    NUM_CLASSES = 26
    BATCH_SIZE = 256
    GRAYSCALE = False


    ##############################
    # Dataset
    ##############################
    ALL_DF   = pd.read_csv(CSV_PATH, index_col=0)
    TRAIN_DF = ALL_DF.sample(frac=0.72, random_state=RANDOM_SEED)
    REST_DF  = ALL_DF.drop(TRAIN_DF.index)
    VALID_DF = REST_DF.sample(frac=0.08/0.28, random_state=RANDOM_SEED)
    TEST_DF  = REST_DF.drop(VALID_DF.index)
    class AFAD_Dataset(Dataset):
        """Custom Dataset for loading AFAD face images"""
        def __init__(self, df, img_dir, transform=None):
            self.img_dir = img_dir
            self.img_paths = df['path'].values
            self.y = df['age'].values
            self.transform = transform

        def __getitem__(self, index):
            img = Image.open(os.path.join(self.img_dir, self.img_paths[index]))
            if self.transform is not None:
                img = self.transform(img)
            label = int(self.y[index])
            return img, label

        def __len__(self):
            return self.y.shape[0]

    custom_transform  = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomCrop((120, 120)), transforms.ToTensor()])
    custom_transform2 = transforms.Compose([transforms.Resize((128, 128)), transforms.CenterCrop((120, 120)), transforms.ToTensor()])
    train_dataset  = AFAD_Dataset(df=TRAIN_DF, img_dir=IMAGE_PATH, transform=custom_transform)
    train2_dataset = AFAD_Dataset(df=TRAIN_DF, img_dir=IMAGE_PATH, transform=custom_transform2)
    valid2_dataset = AFAD_Dataset(df=VALID_DF, img_dir=IMAGE_PATH, transform=custom_transform2)
    test2_dataset  = AFAD_Dataset(df=TEST_DF,  img_dir=IMAGE_PATH, transform=custom_transform2)
    train_loader   = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    train2_loader  = DataLoader(dataset=train2_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    valid2_loader  = DataLoader(dataset=valid2_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test2_loader   = DataLoader(dataset=test2_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    ##############################
    # MODEL
    ##############################
    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes, grayscale):
            self.num_classes = num_classes
            self.inplanes = 64
            if grayscale: in_dim = 1
            else: in_dim = 3
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(4)
            self.fc = nn.Linear(512, 1)
            self.bi = nn.Parameter(math.sqrt(1.0)*torch.ones(self.num_classes-2).float())

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, (2. / n)**.5)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            #
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            #
            x = x.view(x.size(0), -1)
            #
            fc = self.fc(x)+1.0*(NUM_CLASSES-2.)/2.
            #
            tmp = torch.zeros(self.num_classes-1).float().to(DEVICE)
            tmp[0] = torch.tensor([0.0]).to(DEVICE)
            for k in range(1, self.num_classes-1):
                tmp[k] = tmp[k-1] + torch.pow(self.bi[k-1],2)
            #
            return fc, tmp

    def resnet(num_classes, grayscale):
        """Constructs a ResNet-34 model."""
        model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=grayscale)
        return model


    ##############################
    # Settings
    ##############################
    def loss_fn(a, b, targets):
        tmpl = torch.cat([torch.zeros(a.shape[0],1).to(DEVICE),-F.logsigmoid(-b+a)],dim=1).gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        tmpr = torch.cat([-F.logsigmoid(b-a),torch.zeros(a.shape[0],1).to(DEVICE)],dim=1).gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        return torch.mean(tmpl+tmpr)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    model = resnet(NUM_CLASSES, GRAYSCALE)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-6)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.1 ** (epoch/NUM_EPOCHS))

    def OT_func(a, y, K, loss):
        #sort (a,y)
        a, idx = torch.sort(a.reshape(-1))
        y = y[idx]
        #all size n1, unique size n2
        n1 = a.shape[0]
        ua = a.unique(sorted=True).to(dtype=torch.float64)
        n2 = ua.shape[0]
        #DP matrix
        L  = torch.zeros(n2, K, dtype=torch.float64).to(DEVICE)
        #
        Ys = y[a==ua[0]]
        for k in range(K):
            L[0,k] = torch.sum(loss[Ys,k])
        #
        for j in range(1,n2):
            Ys = y[a==ua[j]]
            for k in range(K):
                L[j,k] = torch.min(L[j-1,:k+1]) + torch.sum(loss[Ys,k])
        #threshold parameters
        t = torch.zeros(K-1, dtype=torch.float64).to(DEVICE)
        #
        I = torch.argmin(L[-1,:-1]).item()
        for k in range(I,K-1): t[k] = 10.**8
        #
        for j in reversed(range(n2-1)):
            J = torch.argmin(L[j,:I+1]).item()
            if I!=J:
                for k in range(J,I):
                    t[k] = (ua[j]+ua[j+1])*0.5
                I = J
        #
        for k in range(I): t[k] = -10.**8
        return t

    def compute_errors(model, data_loader, labeling, train=None, t_Z=None, t_A=None, t_S=None):
        MZE, MAE, MSE, num_examples = 0., 0., 0., 0
        if labeling=='RL' or (labeling=='OT' and train==True):
            L_Z = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float).to(DEVICE)
            for j, k in product(range(NUM_CLASSES),range(NUM_CLASSES)):
                if j!=k: L_Z[j,k] = 1.
            L_A = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float).to(DEVICE)
            for j, k in product(range(NUM_CLASSES),range(NUM_CLASSES)):
                L_A[j,k] = abs(j-k)
            L_S = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float).to(DEVICE)
            for j, k in product(range(NUM_CLASSES),range(NUM_CLASSES)):
                L_S[j,k] = (j-k)**2
            #
            alla = torch.tensor([], dtype=torch.float).to(DEVICE)
            ally = torch.tensor([], dtype=torch.long).to(DEVICE)
        for i, (features, targets) in enumerate(data_loader):
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            a, b = model(features)
            num_examples += targets.size(0)
            #
            if labeling=='RL':
                tmp1 = b-a
                tmp2 = torch.zeros(tmp1.shape[0], NUM_CLASSES).float().to(DEVICE)
                for k in range(1,NUM_CLASSES): tmp2[:,k] = tmp2[:,k-1] - tmp1[:,k-1]
                prob = tmp2.softmax(dim=1)
                #
                predicts_Z = torch.argmin(torch.mm(prob, L_Z), dim=1)
                predicts_A = torch.argmin(torch.mm(prob, L_A), dim=1)
                predicts_S = torch.argmin(torch.mm(prob, L_S), dim=1)
                #
                MZE += torch.sum(predicts_Z != targets)
                MAE += torch.sum(torch.abs(predicts_A - targets))
                MSE += torch.sum((predicts_S - targets)**2)
            if labeling=='ST':
                predicts = torch.sum(a-b>=0., 1)
                #
                MZE += torch.sum(predicts != targets)
                MAE += torch.sum(torch.abs(predicts - targets))
                MSE += torch.sum((predicts - targets)**2)
            if labeling=='OT' and train==False:
                predicts_Z = torch.sum(a-t_Z>=0., 1)
                predicts_A = torch.sum(a-t_A>=0., 1)
                predicts_S = torch.sum(a-t_S>=0., 1)
                #
                MZE += torch.sum(predicts_Z != targets)
                MAE += torch.sum(torch.abs(predicts_A - targets))
                MSE += torch.sum((predicts_S - targets)**2)
            if labeling=='OT' and train==True:
                alla = torch.cat([alla, a])
                ally = torch.cat([ally, targets])
        if labeling=='OT' and train==True:
            t_Z = OT_func(alla, ally, NUM_CLASSES, L_Z)
            t_A = OT_func(alla, ally, NUM_CLASSES, L_A)
            t_S = OT_func(alla, ally, NUM_CLASSES, L_S)
            #
            predicts_Z = torch.sum(alla-t_Z>=0., 1)
            predicts_A = torch.sum(alla-t_A>=0., 1)
            predicts_S = torch.sum(alla-t_S>=0., 1)
            #
            MZE = torch.sum(predicts_Z != ally)
            MAE = torch.sum(torch.abs(predicts_A - ally))
            MSE = torch.sum((predicts_S - ally)**2)
        MZE = MZE.float() / num_examples
        MAE = MAE.float() / num_examples
        MSE = MSE.float() / num_examples
        if labeling=='RL':
            return MZE, MAE, MSE
        if labeling=='ST':
            return MZE, MAE, MSE
        if labeling=='OT' and train==True:
            return MZE, MAE, MSE, t_Z, t_A, t_S
        if labeling=='OT' and train==False:
            return MZE, MAE, MSE


    ##############################
    # Validation Phase
    ##############################
    for epoch in range(NUM_EPOCHS):
        # TRAINING
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            # FORWARD AND BACK PROP
            a, b = model(features)
            loss = loss_fn(a, b, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss.item())
        #scheduler.step()
        # EVALUATION
        best_loss = 10.**8
        best_RL_Z, best_RL_A, best_RL_S = 10.**8, 10.**8, 10.**8
        best_ST_Z, best_ST_A, best_ST_S = 10.**8, 10.**8, 10.**8
        best_OT_Z, best_OT_A, best_OT_S = 10.**8, 10.**8, 10.**8
        model.eval()
        with torch.set_grad_enabled(False):
            #surrogate risk
            tr_loss, va_loss, te_loss, tr_n, va_n, te_n = 0., 0., 0., 0, 0, 0
            for features, targets in train2_loader:
                features, targets = features.to(DEVICE), targets.to(DEVICE); n = len(targets)
                a, b = model(features); tr_loss += loss_fn(a, b, targets)*n; tr_n += n
            for features, targets in valid2_loader:
                features, targets = features.to(DEVICE), targets.to(DEVICE); n = len(targets)
                a, b = model(features); va_loss += loss_fn(a, b, targets)*n; va_n += n
            for features, targets in test2_loader:
                features, targets = features.to(DEVICE), targets.to(DEVICE); n = len(targets)
                a, b = model(features); te_loss += loss_fn(a, b, targets)*n; te_n += n
            tr_loss = tr_loss/tr_n; va_loss = va_loss/va_n; te_loss = te_loss/te_n
            #
            print(tr_loss, va_loss, te_loss)
            s_LO = '%f,%f,%f' % (tr_loss, va_loss, te_loss)
            with open(LOGFILE_LO, 'a') as f: f.write('%s\n' % s_LO)
            #if va_loss<best_loss: best_loss = va_loss; torch.save(model.state_dict(), os.path.join(PATH, 'best_loss.pt'))
            #task risk with RL labeling
            tr_RL_Z, tr_RL_A, tr_RL_S = compute_errors(model, train2_loader, 'RL')
            va_RL_Z, va_RL_A, va_RL_S = compute_errors(model, valid2_loader, 'RL')
            te_RL_Z, te_RL_A, te_RL_S = compute_errors(model, test2_loader,  'RL')
            #
            print(tr_RL_Z, tr_RL_A, tr_RL_S, va_RL_Z, va_RL_A, va_RL_S, te_RL_Z, te_RL_A, te_RL_S)
            s_RL = '%f,%f,%f,%f,%f,%f,%f,%f,%f' % (tr_RL_Z, tr_RL_A, tr_RL_S, va_RL_Z, va_RL_A, va_RL_S, te_RL_Z, te_RL_A, te_RL_S)
            with open(LOGFILE_RL, 'a') as f: f.write('%s\n' % s_RL)
            #if va_RL_Z<best_RL_Z: best_RL_Z = va_RL_Z; torch.save(model.state_dict(), os.path.join(PATH, 'best_RL_Z.pt'))
            #if va_RL_A<best_RL_A: best_RL_A = va_RL_A; torch.save(model.state_dict(), os.path.join(PATH, 'best_RL_A.pt'))
            #if va_RL_S<best_RL_S: best_RL_S = va_RL_S; torch.save(model.state_dict(), os.path.join(PATH, 'best_RL_S.pt'))
            #task risk with ST labeling
            tr_ST_Z, tr_ST_A, tr_ST_S = compute_errors(model, train2_loader, 'ST')
            va_ST_Z, va_ST_A, va_ST_S = compute_errors(model, valid2_loader, 'ST')
            te_ST_Z, te_ST_A, te_ST_S = compute_errors(model, test2_loader,  'ST')
            #
            print(tr_ST_Z, tr_ST_A, tr_ST_S, va_ST_Z, va_ST_A, va_ST_S, te_ST_Z, te_ST_A, te_ST_S)
            s_ST = '%f,%f,%f,%f,%f,%f,%f,%f,%f' % (tr_ST_Z, tr_ST_A, tr_ST_S, va_ST_Z, va_ST_A, va_ST_S, te_ST_Z, te_ST_A, te_ST_S)
            with open(LOGFILE_ST, 'a') as f: f.write('%s\n' % s_ST)
            #if va_ST_Z<best_ST_Z: best_ST_Z = va_ST_Z; torch.save(model.state_dict(), os.path.join(PATH, 'best_ST_Z.pt'))
            #if va_ST_A<best_ST_A: best_ST_A = va_ST_A; torch.save(model.state_dict(), os.path.join(PATH, 'best_ST_A.pt'))
            #if va_ST_S<best_ST_S: best_ST_S = va_ST_S; torch.save(model.state_dict(), os.path.join(PATH, 'best_ST_S.pt'))
            #task risk with OT labeling
            _, _, _,  t_Z,  t_A,  t_S = compute_errors(model, train2_loader, 'OT', True)
            tr_OT_Z, tr_OT_A, tr_OT_S = compute_errors(model, train2_loader, 'OT', False, t_Z, t_A, t_S)
            va_OT_Z, va_OT_A, va_OT_S = compute_errors(model, valid2_loader, 'OT', False, t_Z, t_A, t_S)
            te_OT_Z, te_OT_A, te_OT_S = compute_errors(model, test2_loader,  'OT', False, t_Z, t_A, t_S)
            #
            print(tr_OT_Z, tr_OT_A, tr_OT_S, va_OT_Z, va_OT_A, va_OT_S, te_OT_Z, te_OT_A, te_OT_S)
            s_OT = '%f,%f,%f,%f,%f,%f,%f,%f,%f' % (tr_OT_Z, tr_OT_A, tr_OT_S, va_OT_Z, va_OT_A, va_OT_S, te_OT_Z, te_OT_A, te_OT_S)
            with open(LOGFILE_OT, 'a') as f: f.write('%s\n' % s_OT)
            #if va_OT_Z<best_OT_Z: best_OT_Z = va_OT_Z; torch.save(model.state_dict(), os.path.join(PATH, 'best_OT_Z.pt'))
            #if va_OT_A<best_OT_A: best_OT_A = va_OT_A; torch.save(model.state_dict(), os.path.join(PATH, 'best_OT_A.pt'))
            #if va_OT_S<best_OT_S: best_OT_S = va_OT_S; torch.save(model.state_dict(), os.path.join(PATH, 'best_OT_S.pt'))
        #bias parameters
        b = b.to('cpu').detach().numpy().copy()
        s = '%f'%b[0]
        for i in range(1,len(b)): s = s+',%f'%b[i]
        with open(LOGFILE_B, 'a') as f: f.write('%s\n' % s)
        #OT parameters for Task-Z
        t_Z = t_Z.to('cpu').detach().numpy().copy()
        s = '%f'%t_Z[0]
        for i in range(1,len(t_Z)): s = s+',%f'%t_Z[i]
        with open(LOGFILE_Z, 'a') as f: f.write('%s\n' % s)
        #OT parameters for Task-A
        t_A = t_A.to('cpu').detach().numpy().copy()
        s = '%f'%t_A[0]
        for i in range(1,len(t_A)): s = s+',%f'%t_A[i]
        with open(LOGFILE_A, 'a') as f: f.write('%s\n' % s)
        #OT parameters for Task-S
        t_S = t_S.to('cpu').detach().numpy().copy()
        s = '%f'%t_S[0]
        for i in range(1,len(t_S)): s = s+',%f'%t_S[i]
        with open(LOGFILE_S, 'a') as f: f.write('%s\n' % s)
