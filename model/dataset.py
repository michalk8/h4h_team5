#!/usr/bin/env python3

import torch
import os
import numpy as np

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
    
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFilter


class MyDataset(Dataset):

    def __init__(self, path, degradations=None):
        self.paths = [os.path.join(path, cls, f)
                      for cls in os.listdir(path)
                      for f in os.listdir(os.path.join(path, cls))]
        self.classes= [cls
                       for cls in os.listdir(path)
                       for f in os.listdir(os.path.join(path, cls))]
        self.class_mapper = {k: i for i, k in enumerate(set(self.classes))}
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180)
            
        ])
        self.degradations = degradations
        print('Number of classes:', len(self.class_mapper))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img = Image.open(self.paths[idx]).convert('RGB')
        y = self.class_mapper[self.classes[idx]]
        img = self.augmentations(img)
        
        if self.degradations:
            for i in self.degradations:
                w = re.split('_', i)
                if w[0] == 'gaussblur':
                    sigma = w[1]
                    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
                elif w[0] == 'jpeg':
                    quality = w[1]
                    out = BytesIO()
                    img.save(out, format='JPEG',quality=quality)
                    img = out.seek(0)    
                    
        totensor = transforms.ToTensor()
        img = totensor(img)
        return img, y


def get_datasets(root_dir, batch_size=16):
    trn_path, tst_path = os.path.join(root_dir, 'train',), os.path.join(root_dir, 'test')
    trn_loader = DataLoader(MyDataset(trn_path), batch_size=batch_size, shuffle=True,
                            pin_memory=True, sampler=None)
    tst_loader = DataLoader(MyDataset(tst_path), batch_size=batch_size, shuffle=False,
                            pin_memory=True, sampler=None)

    return trn_loader, tst_loader


if __name__ == '__main__':
    dataset, _ = get_datasets('./../dataset_rem_lr')
    for img, _ in dataset:
        Image.fromarray(img.detach().cpu().numpy()[0, ..., :3]).show()
        break
