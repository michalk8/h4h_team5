#!/usr/bin/env python3

import torch
import os
import numpy as np
import re

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
    
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from imbalanced_sampler import ImbalancedDatasetSampler
from PIL import ImageFilter


class MyDataset(Dataset):

    def __init__(self, path, degradations=None, resize=None):
        self.paths = [os.path.join(path, cls, f)
                      for cls in os.listdir(path)
                      for f in os.listdir(os.path.join(path, cls))]
        self.classes = [cls
                        for cls in os.listdir(path)
                        for f in os.listdir(os.path.join(path, cls))]
        self.class_mapper = {k: i for i, k in enumerate(set(self.classes))}
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180, fill=255)
            
        ])
        self.degradations = degradations
        self.resize = resize
        print('Number of classes:', len(self.class_mapper))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img = Image.open(self.paths[idx]).convert('RGB')
        y = self.class_mapper[self.classes[idx]]
        
        if self.resize:
            resize_transform = transforms.Resize(self.resize)
            img = resize_transform(img)
            
        img = self.augmentations(img)
        
        if self.degradations:
            for i in self.degradations:
                w = re.split('_', i)
                if w[0] == 'gaussblur' and w[1] != '0':
                    sigma = int(w[1])
                    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
                elif w[0] == 'jpeg' and w[1] != '0':
                    quality = int(w[1])
                    out = BytesIO()
                    img.save(out, format='JPEG', quality=quality)
                    out.seek(0)
                    img = Image.open(out)

        totensor = transforms.ToTensor()
        img = totensor(img)

        return img, y


def get_datasets(root_dir, batch_size=16):
    trn_path, tst_path = os.path.join(root_dir, 'train',), os.path.join(root_dir, 'test')
    trn_dataset = MyDataset(trn_path)
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, sampler=ImbalancedDatasetSampler(trn_dataset,
                                                                              callback_get_label=lambda dataset, ix: dataset.classes[ix]))
    tst_loader = DataLoader(MyDataset(tst_path), batch_size=batch_size, shuffle=False,
                            pin_memory=True, sampler=None)

    return trn_loader, tst_loader


if __name__ == '__main__':
    dataset, _ = get_datasets('./../../dataset_rem_lr')
    for img, _ in dataset:
        img = img.numpy()[0]
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).show()
        break
