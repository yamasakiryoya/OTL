##############################
# coding: utf-8
# use like > python morph2-NLL-Time.py --cuda 0
##############################
# Imports
##############################
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

torch.backends.cudnn.deterministic = True

TRAIN_CSV_PATH = './morph2_train.csv'
TEST_CSV_PATH  = './morph2_test.csv'
VALID_CSV_PATH = './morph2_valid.csv'
IMAGE_PATH = '../datasets/morph2-aligned'


for RANDOM_SEED in [0]:
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
    NUM_WORKERS = args.numworkers
    PATH = "threshold/NLL-Time"
    if not os.path.exists(PATH): os.makedirs(PATH)
    LOGFILE = os.path.join(PATH, 'training.log')
    header = []
    header.append('PyTorch Version: %s' % torch.__version__)
    header.append('Random Seed: %s' % RANDOM_SEED)
    header.append('Output Path: %s' % PATH)
    header.append('Script: %s' % sys.argv[0])
    with open(LOGFILE, 'w') as f:
        for entry in header:
            print(entry)
            f.write('%s\n' % entry)
            f.flush()


    ##############################
    # Settings
    ##############################
    # Hyperparameters
    learning_rate = 0.001
    NUM_EPOCHS = 100

    # Architecture
    NUM_CLASSES = 55
    BATCH_SIZE = 256
    GRAYSCALE = False


    ##############################
    # Dataset
    ##############################
    class MORPH2_Dataset(Dataset):
        """Custom Dataset for loading MORPH2 face images"""
        def __init__(self, csv_path, img_dir, transform=None):
            df = pd.read_csv(csv_path, index_col=0)
            self.img_dir = img_dir
            self.csv_path = csv_path
            self.img_names = df['file'].values
            self.y = df['age'].values
            self.transform = transform

        def __getitem__(self, index):
            img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
            if self.transform is not None:
                img = self.transform(img)
            label = int(self.y[index])
            return img, label

        def __len__(self):
            return self.y.shape[0]

    custom_transform  = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomCrop((120, 120)), transforms.ToTensor()])
    custom_transform2 = transforms.Compose([transforms.Resize((128, 128)), transforms.CenterCrop((120, 120)), transforms.ToTensor()])
    train_dataset = MORPH2_Dataset(csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH, transform=custom_transform)
    valid_dataset = MORPH2_Dataset(csv_path=VALID_CSV_PATH, img_dir=IMAGE_PATH, transform=custom_transform2)
    test_dataset  = MORPH2_Dataset(csv_path=TEST_CSV_PATH,  img_dir=IMAGE_PATH, transform=custom_transform2)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    valid_loader  = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader   = DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


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
            self.b0 = nn.Parameter(torch.tensor([0.]), requires_grad=False)
            self.bi = nn.Parameter(torch.arange(1,self.num_classes-1).float())

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
            fc = self.fc(x)
            probas = torch.sigmoid(fc - torch.cat((self.b0,self.bi)))
            return fc, torch.cat((self.b0,self.bi)), probas

    def resnet(num_classes, grayscale):
        """Constructs a ResNet-34 model."""
        model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=grayscale)
        return model


    ##############################
    # Settings
    ##############################
    def loss_fn(g, b, targets):
        tmpb = torch.cat((torch.tensor([-10.**8]).to(DEVICE), b, torch.tensor([10.**8]).to(DEVICE))).reshape(-1,1)
        tmpr = torch.sigmoid(tmpb[targets+1]-g)
        tmpl = torch.sigmoid(tmpb[targets]-g)
        return torch.mean(-torch.log(tmpr-tmpl+.1**8))

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    model = resnet(NUM_CLASSES, GRAYSCALE)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def compute_errors(model, data_loader, labeling, train=None, V_A=None):
        restime, MAE, num_examples = 0., 0., 0
        if labeling=='SMB' or (labeling=='ROT' and train==True):
            L_A = torch.zeros(NUM_CLASSES,NUM_CLASSES, dtype=torch.float).to(DEVICE)
            for j in range(NUM_CLASSES):
                for k in range(NUM_CLASSES):
                    L_A[j,k] = abs(j-k)
        if labeling=='ROT' and train==True:
            allg = torch.tensor([], dtype=torch.float).to(DEVICE)
            ally = torch.tensor([], dtype=torch.long).to(DEVICE)
        for i, (features, targets) in enumerate(data_loader):
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            g, b, probas = model(features)
            num_examples += targets.size(0)
            #
            if labeling=='SMB':
                Mprobas = torch.cat([torch.ones(probas.size(0), 1).to(DEVICE), probas], 1) - torch.cat([probas, torch.zeros(probas.size(0), 1).to(DEVICE)], 1)
                predicts_A = torch.argmin(torch.mm(Mprobas, L_A), 1)
                MAE += torch.sum(torch.abs(predicts_A - targets))
            if labeling=='CT':
                predicts = torch.sum(g-b > 0., 1)
                #
                MAE += torch.sum(torch.abs(predicts - targets))
            if labeling=='ROT' and train==True:
                allg = torch.cat((allg, g))
                ally = torch.cat((ally, targets))
            if labeling=='ROT' and train==False:
                predicts_A = torch.sum(g-V_A > 0., 1)
                MAE += torch.sum(torch.abs(predicts_A - targets))
        if labeling=='ROT' and train==True:
            restime -= time.time()
            allg, indeces = torch.sort(allg,0)
            ally = ally[indeces.reshape(-1)]
            #
            M_A = torch.zeros(NUM_CLASSES-1, num_examples+1, dtype=torch.float).to(DEVICE)
            for i in range(num_examples):
                M_A[:,i+1] = M_A[:,i] + L_A[:-1,ally[i]] - L_A[1:,ally[i]]
            tmp1 = torch.argmin(M_A, 1)-1; tmp1[tmp1<0] = 0
            tmp2 = torch.argmin(M_A, 1);   tmp2[tmp2==num_examples] = num_examples-1
            V_A = (allg[tmp1,0] + allg[tmp2,0])/2.
            restime += time.time()
            #
            predicts_A = torch.sum(allg-V_A > 0., 1)
            MAE = torch.sum(torch.abs(predicts_A - ally))
        MAE  = MAE.float() / num_examples
        if labeling=='SMB':
            return MAE
        if labeling=='CT':
            return MAE, int(torch.equal(b, torch.sort(b)[0]))
        if labeling=='ROT' and train==True:
            return MAE, int(torch.equal(b, torch.sort(b)[0])), int(torch.equal(V_A, torch.sort(V_A)[0])), V_A, restime
        if labeling=='ROT' and train==False:
            return MAE


    ##############################
    # Validation Phase
    ##############################
    training_time = 0.
    CT_validation_time = 0.
    SMB_validation_time = 0.
    ROT_validation_time = 0.
    ROT_training_time = 0.
    Best_SMB_A = 10.**8
    Best_CT_A  = 10.**8
    Best_ROT_A = 10.**8

    for epoch in range(NUM_EPOCHS):
        training_time -= time.time()
        # TRAINING
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            # FORWARD AND BACK PROP
            g, b, _ = model(features)
            loss = loss_fn(g, b, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # LOGGING
            if not batch_idx % 50:
                s = ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' % (epoch+1, NUM_EPOCHS, batch_idx, len(train_dataset)//BATCH_SIZE, loss))
                print(s)
                with open(LOGFILE, 'a') as f: f.write('%s\n' % s)
        training_time += time.time()
        # EVALUATION
        model.eval()
        with torch.set_grad_enabled(False):
            SMB_validation_time -= time.time()
            SMB_A = compute_errors(model, valid_loader, 'SMB')
            SMB_validation_time += time.time()
            #
            CT_validation_time -= time.time()
            CT_A, b_ord = compute_errors(model, valid_loader, 'CT')
            CT_validation_time += time.time()
            #
            _, _, va_ord, V_A, restime = compute_errors(model, train_loader, 'ROT', True)
            ROT_training_time += restime
            #
            ROT_validation_time -= time.time()
            ROT_A = compute_errors(model, valid_loader, 'ROT', False, V_A)
            ROT_validation_time += time.time()
        # SAVE BEST MODELS
        if SMB_A <= Best_SMB_A: Best_SMB_A, Best_SMB_A_ep = SMB_A, epoch; torch.save(model.state_dict(), os.path.join(PATH, 'Best-SMB-A.pt'))
        if CT_A  <= Best_CT_A:  Best_CT_A,  Best_CT_A_ep  = CT_A,  epoch; torch.save(model.state_dict(), os.path.join(PATH, 'Best-CT-A.pt'))
        if ROT_A <= Best_ROT_A: Best_ROT_A, Best_ROT_A_ep = ROT_A, epoch; torch.save(model.state_dict(), os.path.join(PATH, 'Best-ROT-A.pt'))
        # SAVE CURRENT/BEST ERRORS/TIME
        s = 'MZE/MAE/RMSE | Current : %.4f/%.4f/%.4f Ep. %d Ord. %d/%d | Best-SMB : %.4f Ep. %d | Best-CT : %.4f Ep. %d | Best-ROT : %.4f Ep. %d' % ( 
            SMB_A, CT_A, ROT_A, epoch, b_ord, va_ord, Best_SMB_A, Best_SMB_A_ep, Best_CT_A, Best_CT_A_ep, Best_ROT_A, Best_ROT_A_ep)
        print(s)
        with open(LOGFILE, 'a') as f: f.write('%s\n' % s)
        #
        s = 'Time: %.4f/%.4f/%.4f/%.4f/%.4f min' % ( 
            training_time, CT_validation_time, SMB_validation_time, ROT_validation_time, ROT_training_time)
        print(s)
        with open(LOGFILE, 'a') as f: f.write('%s\n' % s)

    ##############################
    # Test Phase
    ##############################
    for labeling in ['SMB', 'CT', 'ROT']:
        for task in ['A']:
            # SAVE BEST ERRORS
            model.load_state_dict(torch.load(os.path.join(PATH, 'Best-%s-%s.pt'%(labeling, task))))
            model.eval()
            with torch.set_grad_enabled(False):
                if labeling=='SMB':
                    tr_MAE = compute_errors(model, train_loader, labeling)
                    va_MAE = compute_errors(model, valid_loader, labeling)
                    te_MAE = compute_errors(model, test_loader,  labeling)
                    #
                    s = 'Best-%s-%s MZE/MAE/RMSE | Train: %.4f | Valid: %.4f | Test: %.4f' % (
                        labeling, task, tr_MAE, va_MAE, te_MAE)
                    print(s)
                    with open(LOGFILE, 'a') as f: f.write('%s\n' % s)
                if labeling=='CT':
                    tr_MAE, b_ord = compute_errors(model, train_loader, labeling)
                    va_MAE, _ = compute_errors(model, valid_loader, labeling)
                    te_MAE, _ = compute_errors(model, test_loader,  labeling)
                    #
                    s = 'Best-%s-%s MZE/MAE/RMSE | Train: %.4f | Valid: %.4f | Test: %.4f, Order | b: %d' % (
                        labeling, task, tr_MAE, va_MAE, te_MAE, b_ord)
                    print(s)
                    with open(LOGFILE, 'a') as f: f.write('%s\n' % s)
                if labeling=='ROT':
                    tr_MAE, b_ord, va_ord, V_A, _ = compute_errors(model, train_loader, labeling, True)
                    va_MAE = compute_errors(model, valid_loader, labeling, False, V_A)
                    te_MAE = compute_errors(model, test_loader,  labeling, False, V_A)
                    #
                    s = 'Best-%s-%s MZE/MAE/RMSE | Train: %.4f | Valid: %.4f | Test: %.4f, Order | b&v: %d/%d' % (
                        labeling, task, tr_MAE, va_MAE, te_MAE, b_ord, va_ord)
                    print(s)
                    with open(LOGFILE, 'a') as f: f.write('%s\n' % s)

    os.remove(os.path.join(PATH, 'Best-SMB-A.pt'))
    os.remove(os.path.join(PATH, 'Best-CT-A.pt'))
    os.remove(os.path.join(PATH, 'Best-ROT-A.pt'))
