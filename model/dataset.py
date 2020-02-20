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

    def __init__(self, path, img_size, is_train=True, degradations=None, resize=None):
        self.paths = [os.path.join(path, cls, f)
                      for cls in os.listdir(path)
                      for f in os.listdir(os.path.join(path, cls))]
        self.classes = [cls
                        for cls in os.listdir(path)
                        for f in os.listdir(os.path.join(path, cls))]
        self.img_size = (img_size, img_size)
        self.class_mapper = {k: i for i, k in enumerate(set(self.classes))}
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180, fill=255)
            
        ])
        self.degradations = degradations
        self.resize = resize
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()
        self.finalize = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225]),
            #transforms.Normalize(mean=[0.8209723354249416, 0.7285539707273201, 0.8370288417509042],
            #                     std=[0.002150924859562229, 0.0032857094678095506, 0.0016802816657651217]),
            transforms.ToPILImage()
        ])

        print('Number of classes:', len(self.class_mapper))
        print('Image size:', img_size)
        print('Degradations:', self.degradations)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img = Image.open(self.paths[idx]).convert('RGB').resize(size=self.img_size, resample=Image.BILINEAR)
        img = self.finalize(img)
        y = self.class_mapper[self.classes[idx]]
        
        if self.resize:
            resize_transform = transforms.Resize(self.resize)
            img = resize_transform(img)

        if self.is_train:
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

        img = self.to_tensor(img)

        return img, y


def get_datasets(root_dir, img_size, degradations, batch_size=16):
    trn_path, tst_path = os.path.join(root_dir, 'train',), os.path.join(root_dir, 'test')
    trn_dataset = MyDataset(trn_path, img_size, is_train=True, degradations=degradations)
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, sampler=ImbalancedDatasetSampler(trn_dataset,
                                                                              callback_get_label=lambda dataset, ix: dataset.classes[ix]))
    tst_dataset = MyDataset(tst_path, img_size, is_train=False)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, sampler=None)

    return trn_loader, tst_loader


if __name__ == '__main__':
    dataset, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=[])
    for img, _ in dataset:
        img = img.numpy()[0]
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).show()
        break
