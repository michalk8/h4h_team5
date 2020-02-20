#!/usr/bin/env python3

import torch
import os

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from imbalanced_sampler import ImbalancedDatasetSampler


class MyDataset(Dataset):

    def __init__(self, path):
        self.paths = [os.path.join(path, cls, f)
                      for cls in os.listdir(path)
                      for f in os.listdir(os.path.join(path, cls))]
        self.classes= [cls
                       for cls in os.listdir(path)
                       for f in os.listdir(os.path.join(path, cls))]
        self.class_mapper = {k: i for i, k in enumerate(set(self.classes))}
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        print('Number of classes:', len(self.class_mapper))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img = Image.open(self.paths[idx]).convert('RGB')
        y = self.class_mapper[self.classes[idx]]

        return self.transforms(img), y


def get_datasets(root_dir, batch_size=16):
    trn_path, tst_path = os.path.join(root_dir, 'train',), os.path.join(root_dir, 'test')
    trn_dataset = MyDataset(trn_path)
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=True, sampler=ImbalancedDatasetSampler(trn_dataset))
    tst_loader = DataLoader(MyDataset(tst_path), batch_size=batch_size, shuffle=False,
                            pin_memory=True, sampler=None)

    return trn_loader, tst_loader


if __name__ == '__main__':
    dataset, _ = get_datasets('/home/michal/dataset_rem_lr')
    for img, _ in dataset:
        print(img.shape)
        Image.fromarray(img.detach().cpu().numpy()[0, ..., :3]).show()
        break
