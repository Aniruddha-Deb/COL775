#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.cuda import is_available
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import pickle

from torch.utils.data import Dataset, DataLoader, Subset

"""
PARAMETERS
"""

lr = 1e-5
dpath = 'data'

splits = {
    'train': [f'{dpath}/data_batch_{i}' for i in range(1,5)],
    'val'  : [f'{dpath}/data_batch_5']
}

DEBUG = False
debug_len = {
    'train': 128,
    'val': 32
}

class Cifar10Dataset(Dataset):
    
    def __init__(self, batch_files):
        
        image_batches = []
        label_batches = []
        
        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                image_batches.append(batch[b'data'])
                label_batches.append(batch[b'labels'])
        
        self.images = torch.tensor(np.vstack(image_batches)).float()
        self.labels = torch.tensor(np.hstack(label_batches)).long()
        
        # image transformations
        mean = self.images.mean(axis=0)
        stdev = self.images.std(axis=0)
        
        self.images = (self.images - mean)/stdev
        
        self.images = self.images.reshape((self.images.shape[0],3,32,32))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])

# modelled after the PyTorch implementation: Have a base normalizer which
# stores the parameters, and subclasses which use it's buffers
class Normalizer(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((num_features, 1)))
        self.bias = nn.Parameter(torch.zeros((num_features, 1)))

        self.running_mean = nn.Parameter(torch.zeros((num_features, 1)), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones((num_features, 1)), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)

    def reset_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_params(self):
        self.weight.fill_(1)
        self.bias.zero_()

    def dim_check(self, x, tgt_dim):
        if (x.dim() != tgt_dim):
            raise ValueError(f'Expected {tgt_dim}D input, got {x.dim()}D input')

class BatchNormalizer(Normalizer):

    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, x):
        # x : (B, C, H, W)
        self.dim_check(x, 4)
        b, c, h, w = x.shape
        x = x.flatten(start_dim=2, end_dim=3)
        # x : (B, C, H*W)

        if self.training:
            mean = x.mean((0,2)).unsqueeze(1) # over all pixels and examples (one mean per channel)
            var = x.var((0,2), unbiased=False).unsqueeze(1)

            # n = self.num_batches_tracked
            # n_new = self.num_batches_tracked+1
            # EMA with momentum of 0.1
            self.running_mean.data = 0.1*mean + 0.9*self.running_mean.data #((self.running_mean.data)/n_new)*n + mean/n_new
            self.running_var.data = 0.1*var + 0.9*self.running_var.data #((self.running_var.data)/(n_new*n_new))*(n*n) + var/(n_new*n_new)
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # print(f'x_max: {x.max():.3f}, x_min: {x.min():.3f}')
        # print(f'mean_max: {mean.max():.3f}, mean_min: {mean.min():.3f}')
        # print(f'var_max: {var.max():.3f}, var_min: {var.min():.3f}')
        norm_x = self.weight * (x - mean)/(torch.sqrt(var + 1e-5)) + self.bias
        # print(f'norm_x_max: {norm_x.max():.3f}, norm_x_min: {norm_x.min():.3f}')

        return norm_x.unflatten(2, (h,w))

class LayerNormalizer(Normalizer):

    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, x):
        # x : (B, C, H, W)
        self.dim_check(x, 4)
        b, c, h, w = x.shape
        x = x.flatten(start_dim=1, end_dim=3)
        # x : (B, C*H*W)

        if self.training:
            mean = x.mean(1) 
            var = x.var(1) 

            self.running_mean = ((self.running_mean)/(self.num_batches_tracked+1))*self.num_batches_tracked + \
                    mean/(self.num_batches_tracked+1)
            self.running_var = ((self.running_var)/(self.num_batches_tracked+1))*self.num_batches_tracked + \
                    var/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        norm_x = self.weight * (x - mean.unsqueeze(1))/(torch.sqrt(var.unsqueeze(1) + 1e-5)) + self.bias

        return norm_x.unflatten(1, (c,h,w))
    
class InstanceNormalizer(Normalizer):

    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, x):
        # x : (B, C, H, W)
        self.dim_check(x, 4)
        b, c, h, w = x.shape
        x = x.flatten(start_dim=2, end_dim=3)
        # x : (B, C, H*W)

        if self.training:
            mean = x.mean(2)
            var = x.var(2) 

            self.running_mean = ((self.running_mean)/(self.num_batches_tracked+1))*self.num_batches_tracked + \
                    mean/(self.num_batches_tracked+1)
            self.running_var = ((self.running_var)/(self.num_batches_tracked+1))*self.num_batches_tracked + \
                    var/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        norm_x = self.weight * (x - mean.unsqueeze(1))/(torch.sqrt(var.unsqueeze(1) + 1e-5)) + self.bias

        return norm_x.unflatten(2, (h,w))

# # TODO
# class BatchInstanceNormalizer(Normalizer):
# 
#     def forward(self, x):
#         self.dim_check(x, 4)
#         
#         out_bn = normalize(
#                 x,
#                 [0], # normalize across axis 0 (batches)
#                 self.running_mean,
#                 self.running_var,
#                 self.weight,
#                 self.bias,
#                 self.training)
#         
#         return out_bn

class GroupNormalizer(Normalizer):

    def __init__(self, num_features, num_groups=16):
        super().__init__(num_features)
        self.num_groups = num_groups

    def forward(self, x):
        # x : (B, C, H, W)
        self.dim_check(x, 4)
        b, c, h, w = x.shape
        # this implementation requires that the number of channels is an even
        # multiple of the number of groups
        g = c/self.num_groups
        x = x.reshape((b,self.num_groups, g*h*w))
        # x : (B, n_g, G*H*W)

        if self.training:
            mean = x.mean(2)
            var = x.var(2) 

            self.running_mean = ((self.running_mean)/(self.num_batches_tracked+1))*self.num_batches_tracked + \
                    mean/(self.num_batches_tracked+1)
            self.running_var = ((self.running_var)/(self.num_batches_tracked+1))*self.num_batches_tracked + \
                    var/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        norm_x = self.weight * (x - mean.unsqueeze(1))/(torch.sqrt(var.unsqueeze(1) + 1e-5)) + self.bias

        return norm_x.reshape((b,c,h,w))

class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, batch_norm_class=nn.BatchNorm2d, downsample=False, in_dim=32*32):
        # TODO add ReLU, BatchNorm etc
        super().__init__()
        
        self.downsample = downsample
        
        self.relu = nn.ReLU()
        
        if self.downsample:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                batch_norm_class(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                batch_norm_class(out_channels)
            )
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2),
                batch_norm_class(out_channels),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                batch_norm_class(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                batch_norm_class(out_channels)
            )
    
    def forward(self, x):
        
        if self.downsample:
            return self.relu(self.downsampler(x) + self.block(x))

        return self.relu(x + self.block(x))

class ResNet(nn.Module):
    
    def __init__(self, n, r, batch_norm_class=nn.BatchNorm2d):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Sequential(*[ResNetBlock(16,16,batch_norm_class=batch_norm_class) for _ in range(n)]),
            ResNetBlock(16,32,batch_norm_class=batch_norm_class,downsample=True,in_dim=32*32),
            nn.Sequential(*[ResNetBlock(32,32,batch_norm_class=batch_norm_class) for _ in range(n-1)]),
            ResNetBlock(32,64,batch_norm_class=batch_norm_class,downsample=True,in_dim=16*16),
            nn.Sequential(*[ResNetBlock(64,64,batch_norm_class=batch_norm_class) for _ in range(n-1)]),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,r)
        )
    
    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    datasets = {split: Cifar10Dataset(batch_files) for split, batch_files in splits.items()}

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    if DEBUG:
        datasets = {split: Subset(dataset, np.arange(debug_len[split])) for split, dataset in datasets.items()}
    
    dataloaders = {split: DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True) for split, dataset in datasets.items()}
    
    model = ResNet(2, 10, batch_norm_class=BatchNormalizer).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    for i in tqdm(range(5)):
        
        curr_train_loss = 0
        curr_val_loss = 0
        
        model.train()
        # training
        for images, labels in dataloaders['train']:
            
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            curr_train_loss += loss.detach().item()

        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            # validation
            for images, labels in dataloaders['val']:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                curr_val_loss += loss.detach().item()
            
        curr_train_loss /= len(dataloaders['train'])
        curr_val_loss /= len(dataloaders['val'])
        
        train_losses.append(curr_train_loss)
        val_losses.append(curr_val_loss)
        
        # if (i%5 == 0):
        print(f"Epoch {i}:")
        print(f"    Train loss : {curr_train_loss}")
        print(f"    Val loss   : {curr_val_loss}")
    
    plt.plot(train_losses)
    plt.plot(val_losses)

