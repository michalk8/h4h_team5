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
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(degrees=180, fill=255)
            
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

        return img, y  # commet y to create the figures


def get_datasets(root_dir, img_size, degradations=[], batch_size=16):
    trn_path, tst_path = os.path.join(root_dir, 'train',), os.path.join(root_dir, 'test')
    trn_dataset = MyDataset(trn_path, img_size, is_train=True, degradations=degradations)
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True)#, #sampler=ImbalancedDatasetSampler(trn_dataset,
                                             #                                 callback_get_label=lambda dataset, ix: dataset.classes[ix]))
    tst_dataset = MyDataset(tst_path, img_size, is_train=False)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, sampler=None)

    return trn_loader, tst_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=[])
    dataset_1, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['gaussblur_2'])
    dataset_2, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['gaussblur_4'])
    dataset_3, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['gaussblur_6'])
    dataset_4, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['gaussblur_8'])
    dataset_5, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['gaussblur_10'])

    dataset_j0, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['jpeg_5'])
    dataset_j1, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['jpeg_20'])
    dataset_j2, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['jpeg_40'])
    dataset_j3, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['jpeg_60'])
    dataset_j4, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['jpeg_80'])
    dataset_j5, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=['jpeg_95'])

    dataset_l0, _ = get_datasets('./../../dataset_rem_lr', 400)
    dataset_l1, _ = get_datasets('./../../dataset_rem_lr', 200)
    dataset_l2, _ = get_datasets('./../../dataset_rem_lr', 100)
    dataset_l3, _ = get_datasets('./../../dataset_rem_lr', 50)
    dataset_l4, _ = get_datasets('./../../dataset_rem_lr', 25)

    fig, axes = plt.subplots(1, 6, figsize=(10, 15), dpi=180)

    for (x, a, b, c, d, e) in zip(dataset, dataset_1, dataset_2, dataset_3, dataset_4, dataset_5):
        data = []
        for d in [x, a, b, c, d, e]:
            img = d.numpy()[0]
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1)
            data.append((255 * img).astype(np.uint8)[:])

        for i, (d, ax) in enumerate(zip(data, np.ravel(axes))):
            ax.imshow(d)
            ax.set_title(f'{i * 2 if i != 0 else None}', color='white', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.show()
        fig.savefig('blurred.png', transparent=True)
        break

    fig, axes = plt.subplots(1, 7, figsize=(10, 15), dpi=180)
    dataset, _ = get_datasets('./../../dataset_rem_lr', 400, degradations=[])
    for (x, o, a, b, c, d, e) in zip(dataset_j0, dataset_j1, dataset_j2, dataset_j3, dataset_j4, dataset_j5, dataset):
        data = []
        for d in [x, o, a, b, c, d, e]:
            img = d.numpy()[0]
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1)
            data.append((255 * img).astype(np.uint8)[:])

        for orig, d, ax in zip([5, 20, 40, 60, 80, 95, 'original'], data, np.ravel(axes)):
            ax.imshow(d)
            ax.set_title(f'{orig}', color='white', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.show()
        fig.savefig('jpeg_quality.png', transparent=True)
        break

    fig, axes = plt.subplots(1, 5, figsize=(10, 15), dpi=180)
    for (a, b, c, d, e) in zip(dataset_l0, dataset_l1, dataset_l2, dataset_l3, dataset_l4):
        data = []
        for d in [a, b, c, d, e]:
            img = d.numpy()[0]
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1)
            data.append((255 * img).astype(np.uint8)[:])

        for orig, d, ax in zip([400, 200, 100, 50, 25], data, np.ravel(axes)):
            ax.imshow(d)
            ax.set_title(f'{orig}px', color='white', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.show()
        fig.savefig('size.png', transparent=True)
        break
