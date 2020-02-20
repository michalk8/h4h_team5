#!/usr/bin/env python3

import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
import argparse

from dataset import get_datasets
from torch.optim import lr_scheduler

N_EPOCHS = 5
N_CLS = 15
BATCH_SIZE = 16


def create_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, N_CLS)

    return model


def main(args):
    device = torch.device('gpu:0') if torch.cuda.is_available() else torch.device('cpu')
    model = create_model().to(device)

    trn_loader, tst_loader = get_datasets(args.dataset_root)
    dataloaders = {'train': trn_loader, 'val': tst_loader}
    dataset_sizes = {x: len(dataloaders[x]) * BATCH_SIZE for x in ['train', 'val']}
    print(dataset_sizes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(N_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
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

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=str)
    parser.add_argument('img_size', type=int)
    parser.add_argument('downsampling', type=str)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--jpeg-quality', type=int, default=None)

    main(parser.parse_args())
