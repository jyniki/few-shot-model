import os
import io
import h5py
import json
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# base-classes for baseline classifier training
class Fewshot_dataset(Dataset):
    def __init__(self, datapath, dataset, split, transform):
        self.transform = transform
        self.dataset= dataset
        self.data, self.labels = self._extract_data_from_hdf5(dataset, datapath, split)
        
    def _extract_data_from_hdf5(self, dataset, datapath, split):
        datapath = os.path.join(datapath, dataset)
        # Load omniglot
        if dataset == 'omniglot':
            classes = []
            with h5py.File(os.path.join(datapath, 'data.hdf5'), 'r') as f_data:
                with open(os.path.join(datapath, 'vinyals_{}_labels.json'.format(split))) as f_labels:
                    labels = json.load(f_labels)
                    for label in labels:
                        img_set, alphabet, character = label
                        classes.append(f_data[img_set][alphabet][character][()])

        # Load mini-imageNet 
        elif dataset == 'miniimagenet' or dataset == 'cub' or dataset == 'tieredimagenet':
            with h5py.File(os.path.join(datapath, split + '_data.hdf5'), 'r') as f:
                datasets = f['datasets']
                classes = [datasets[k][()] for k in datasets.keys()]
                labels = [np.repeat([i], len(datasets[k][()])) for i, k in enumerate(datasets.keys())]
                self.n_classes = len(labels)
                self.label = np.concatenate(labels) 
        
        else:
            raise ValueError("No such dataset available.")
        
        data = np.concatenate(classes)
        labels = self.label
        return data, labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.dataset == 'cub' or self.dataset == 'tieredimagenet':
            image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        else:
            image = Image.fromarray(self.data[index])

        image = self.transform(image)
        label = torch.tensor(self.labels[index])
        return image, label
    
    def convert_raw(self, x):
        norm_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
        std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
        return x * std + mean    

# self-superviesed 
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_fewshot_dataset(datapath, dataset, split='train', **kwargs):
    # image resize and augmentation
    image_size = 28 if dataset == 'omniglot' else 84
    
    # Get transform
    norm_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**norm_params)
    augment = kwargs.get('augment')

    if augment == 'crop':
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.RandomCrop(image_size, padding=8),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,])

    elif augment == 'resize':
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,])

    elif augment == 'aug':
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) # 调整亮度、对比度、饱和度和色相
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ToTensor(),
                                        normalize,])
    
    elif augment == 'aug_co':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augmentation = [
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform = TwoCropsTransform(transforms.Compose(augmentation))
    
    else:
        if dataset == 'cub':
            transform = transforms.Compose([transforms.Resize(image_size), 
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(), 
                                            normalize,])

        else:
            transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize,])

    dataset = Fewshot_dataset(datapath, dataset, split, transform)
    return dataset

def get_standard_loader(dataset, batchsize, shuffle=True, num_workers=0):
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return dataloader

def get_meta_loader(dataset, n_batch, ways, shots, query_shots, batch_size, num_workers=0):
    meta_sampler = CategoriesSampler(dataset.label, n_batch, ways, shots + query_shots, ep_per_batch=batch_size)
    meta_dataloader = DataLoader(dataset, batch_sampler=meta_sampler, num_workers=num_workers, pin_memory=True)
    return meta_dataloader

class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch
        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls, replace=False)
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)

def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query

def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label



