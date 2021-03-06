#!/usr/bin/env python3

import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
import numpy as np
import argparse
import pickle

from dataset import get_datasets
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
#from my_metrics import *
from time import time

N_EPOCHS = 10
N_CLS = 15
BATCH_SIZE = 16


def create_model():
    model = models.resnet34(pretrained=True)
    # model = models.resnext50_32x4d(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, N_CLS)

    return model


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = create_model().to(device)
    
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    trn_loader, tst_loader = get_datasets(args.dataset_root, args.img_size, args.degradations)
    dataloaders = {'train': trn_loader, 'val': tst_loader}
    dataset_sizes = {x: len(dataloaders[x]) * BATCH_SIZE for x in ['train', 'val']}
    print(dataset_sizes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    cmats = {'train': [], 'val': []}
    losses = {'train': [], 'val': []}
    accs = {'train': [], 'val': []}

    prec = {'train': [], 'val': []}
    rec = {'train': [], 'val': []}
    f1 = {'train': [], 'val': []}
    
    prec_avg = {'train': [], 'val': []}
    rec_avg = {'train': [], 'val': []}
    f1_avg = {'train': [], 'val': []}

    b_loss = {'train': [], 'val': []}
    b_accs = {'train': [], 'val': []}

    for epoch in range(N_EPOCHS):
        y_trues, y_preds = [], []
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print('TRAIN')
            else:
                model.eval()
                print('EVAL')

            running_loss = 0.0
            running_corrects = 0

            start = time()
            for cnt, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                y_trues.extend(list(labels.cpu().numpy()))
                y_preds.extend(list(preds.cpu().numpy()))

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if cnt % 10:
                    l = (loss.item() * inputs.size(0))
                    a = (torch.sum(preds == labels.data).item() / inputs.size(0))
                    b_loss[phase].append(l)
                    b_accs[phase].append(a)

            if phase == 'train':
                scheduler.step()

            y_trus = np.array(y_trues)
            y_pres = np.array(y_preds)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(float(epoch_loss))
            accs[phase].append(float(epoch_acc))
            cmats[phase].append(confusion_matrix(y_trus, y_pres))
            precision, recall, fscore, support = precision_recall_fscore_support(y_trus, y_pres, )
            precision_avg, recall_avg, fscore_avg, support_avg = precision_recall_fscore_support(y_trus, y_pres, average='weighted')
            
            prec[phase].append(precision)
            rec[phase].append(recall)
            f1[phase].append(fscore)

            prec_avg[phase].append(precision_avg)
            rec_avg[phase].append(recall_avg)
            f1_avg[phase].append(fscore_avg)

            print('{} - {} Loss: {:.4f} Acc: {:.4f}'.format(time() - start,

                phase, epoch_loss, epoch_acc))

    result = {'cmats': cmats, 'losses': losses, 'accs': accs, 'b_loss': b_loss, 'b_acc': b_accs,
              'prec': prec, 'rec': rec, 'f1': f1, 'prec_avg': prec_avg, 'rec_avg': rec_avg, 'f1_avg':f1_avg}
    with open('result_{}_{}.pickle'.format(args.img_size, '_'.join(args.degradations)), 'wb') as fout:
        pickle.dump(result, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=str)
    parser.add_argument('img_size', type=int)
    parser.add_argument('degradations', type=str, nargs='*')

    main(parser.parse_args())
