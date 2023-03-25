#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy
import torchmetrics.functional as tmf

import pickle

from torch.utils.data import Dataset, DataLoader, Subset

DEBUG = False
debug_len = {
    'train': 512,
    'val': 128
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

    # num_features: C
    # although this parameter is unused for channel-invariant filters (eg LayerNormalizer)
    def __init__(self, num_features, track_running_stats=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((num_features, 1)))
        self.bias = nn.Parameter(torch.zeros((num_features, 1)))
        self.eps = 1e-5
        self.momentum = 0.1

        self.tracking_running_stats = track_running_stats
        if self.tracking_running_stats:
            self.running_mean = nn.Parameter(torch.zeros((num_features, 1)), requires_grad=False)
            self.running_var = nn.Parameter(torch.ones((num_features, 1)), requires_grad=False)
            self.num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)

    def reset_stats(self):
        if self.tracking_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_params(self):
        self.weight.fill_(1)
        self.bias.zero_()

    def update_stats(self, mean, var):
        if self.tracking_running_stats:
            self.running_mean.data = self.momentum*mean + (1-self.momentum)*self.running_mean.data
            self.running_var.data = self.momentum*var + (1-self.momentum)*self.running_var.data
            self.num_batches_tracked += 1

    def dim_check(self, x, tgt_dim):
        if (x.dim() != tgt_dim):
            raise ValueError(f'Expected {tgt_dim}D input, got {x.dim()}D input')

class BatchNormalizer(Normalizer):

    def __init__(self, num_features, track_running_stats=True):
        super().__init__(num_features, track_running_stats=track_running_stats)

    def _transform(self, x):
        x = x.flatten(start_dim=2, end_dim=3)
        # x : (B, C, H*W)

        if self.tracking_running_stats and not self.training:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = x.mean((0,2)).unsqueeze(1) # over all pixels and examples (one mean per channel)
            var = x.var((0,2), unbiased=False).unsqueeze(1)
            self.update_stats(mean, var)

        return (x - mean)/torch.sqrt(var + self.eps)

    def forward(self, x):
        # x : (B, C, H, W)
        self.dim_check(x, 4)
        b, c, h, w = x.shape
        x_hat = self._transform(x)

        norm_x = self.weight * x_hat + self.bias

        return norm_x.unflatten(2, (h,w))

class InstanceNormalizer(Normalizer):

    def __init__(self, num_features, track_running_stats=False):
        super().__init__(num_features, track_running_stats=track_running_stats)

    def _transform(self, x):
        # x : (B, C, H, W)
        x = x.flatten(start_dim=2, end_dim=3)
        # x : (B, C, H*W)

        if self.tracking_running_stats and not self.training:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = x.mean(2).unsqueeze(2) # over all pixels and examples (one mean per channel)
            var = x.var(2, unbiased=False).unsqueeze(2)
            self.update_stats(mean, var)

        return (x - mean)/torch.sqrt(var + self.eps)

    def forward(self, x):
        self.dim_check(x, 4)
        b, c, h, w = x.shape

        x_hat = self._transform(x)
        norm_x = self.weight * x_hat + self.bias

        return norm_x.unflatten(2, (h,w))

class BatchInstanceNormalizer(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor([0.5]))

        # BIN uses a BatchNormalizer that tracks running statistics
        self.bnorm = BatchNormalizer(num_features)
        self.inorm = InstanceNormalizer(num_features)

        self.weight = nn.Parameter(torch.ones((num_features, 1)))
        self.bias = nn.Parameter(torch.zeros((num_features, 1)))

    def forward(self, x):
        # x : (B, C, H, W)
        self.bnorm.dim_check(x, 4)
        b, c, h, w = x.shape
        x_bn = self.bnorm._transform(x)
        x_in = self.inorm._transform(x)

        x_bin = self.weight * (self.rho * x_bn + (1 - self.rho) * x_in) + self.bias

        return x_bin.unflatten(2, (h,w))

class LayerNormalizer(Normalizer):

    def __init__(self, num_features):
        super().__init__(1)

    def _transform(self, x):
        # x : (B, C, H, W)
        x = x.flatten(start_dim=1, end_dim=3)
        # x : (B, C*H*W)

        # layer norm uses the current mean and variance and doesn't memorize
        # old means and variances.
        mean = x.mean(1).unsqueeze(1) # over all examples in batch
        var = x.var(1, unbiased=False).unsqueeze(1)

        return (x - mean)/torch.sqrt(var + self.eps)

    def forward(self, x):
        self.dim_check(x, 4)
        b, c, h, w = x.shape

        x_hat = self._transform(x)
        norm_x = self.weight * x_hat + self.bias

        return norm_x.unflatten(1, (c,h,w))

class GroupNormalizer(Normalizer):

    def __init__(self, num_features, num_groups=16):
        super().__init__(num_groups)
        self.num_groups = num_groups

    def forward(self, x):
        # x : (B, C, H, W)
        self.dim_check(x, 4)
        b, c, h, w = x.shape
        g = c//self.num_groups
        x = x.reshape((b,self.num_groups, g*h*w))
        # x : (B, n_g, G*H*W)

        mean = x.mean(2).unsqueeze(2) # over all groups
        var = x.var(2, unbiased=False).unsqueeze(2)

        norm_x = self.weight * (x - mean)/(torch.sqrt(var + self.eps)) + self.bias

        return norm_x.reshape((b,c,h,w))

class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, batch_norm_class=nn.BatchNorm2d, downsample=False, in_dim=32*32):
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
            batch_norm_class(16),
            nn.ReLU(),
            nn.Sequential(*[ResNetBlock(16,16,batch_norm_class=batch_norm_class) for _ in range(n)]),
            ResNetBlock(16,32,downsample=True,in_dim=32*32),
            nn.Sequential(*[ResNetBlock(32,32,batch_norm_class=batch_norm_class) for _ in range(n-1)]),
            ResNetBlock(32,64,batch_norm_class=batch_norm_class,downsample=True,in_dim=16*16),
            nn.Sequential(*[ResNetBlock(64,64,batch_norm_class=batch_norm_class) for _ in range(n-1)]),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,r)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, optimizer, scheduler, dataloaders, loss_fn, 
                max_epochs=100, early_stopping=True, max_patience=5, clip_rho=False):
    train_losses = []
    val_losses = []
    best_val_loss = 10000
    patience = 0
    best_model = None

    bin_gates = []
    if clip_rho:
        bin_gates = [p for p in model.parameters() if getattr(p, 'rho', False)] 

    for i in tqdm(range(max_epochs)):
        
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

            # hacky, but no cleaner option
            if clip_rho:
                for gate in bin_gates:
                    gate.data.clamp_(min=0, max=1)
            
            curr_train_loss += loss.detach().item()

        curr_train_loss /= len(dataloaders['train'])
        train_losses.append(curr_train_loss)
        if scheduler:
            scheduler.step()
        
        if 'val' in dataloaders:
            model.eval()
            with torch.no_grad():
                # validation
                for images, labels in dataloaders['val']:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    curr_val_loss += loss.detach().item()
            curr_val_loss /= len(dataloaders['val'])
            val_losses.append(curr_val_loss)

            if early_stopping:
                if curr_val_loss >= best_val_loss:
                    patience += 1
                    if patience >= max_patience:
                        print(f"Early stopping at epoch {i+1}")
                        break
                else:
                    patience = 0 
                    best_val_loss = curr_val_loss
                    best_model = copy.deepcopy(model)
        
        if (i%5 == 0):
            print(f"Epoch {i+1}:")
            print(f"    Train loss : {curr_train_loss}")
            if 'val' in dataloaders:
                print(f"    Val loss   : {curr_val_loss}")
                # print(f"    Patience   : {patience}")
    
    return best_model, (train_losses, val_losses)

def eval_model(model, dataloaders):

    print("Evaluating...")
    model.eval()
    for dl_name, dl in dataloaders.items():
        all_preds = []
        all_true = []
        for images, labels in dl:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_true.append(labels)

        all_preds = torch.hstack(all_preds).cpu()
        all_true = torch.hstack(all_true).cpu()

        acc = tmf.accuracy(all_preds, all_true, task='multiclass', num_classes=10)
        f1m = tmf.f1_score(all_preds, all_true, task='multiclass', num_classes=10, average='micro')
        f1M = tmf.f1_score(all_preds, all_true, task='multiclass', num_classes=10, average='macro')

        print(f'{dl_name}:')
        print(f'    Accuracy : {acc:.3f}')
        print(f'    F1 micro : {f1m:.3f}')
        print(f'    F1 macro : {f1M:.3f}')

def q1_part2(cifar10_path):
    splits = {'train': [f'{cifar10_path}/data_batch_{i}' for i in range(1,6)]}
    datasets = {split: Cifar10Dataset(batch_files) for split, batch_files in splits.items()}

    n_epochs = 100
    if DEBUG:
        datasets = {split: Subset(dataset, np.arange(debug_len[split])) for split, dataset in datasets.items()}
        n_epochs = 5

    dataloaders = {split: DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True) for split, dataset in datasets.items()}
    model = ResNet(2, 10).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    best_model, (train_losses, val_losses) = train_model(model, optimizer, scheduler, dataloaders, loss_fn, max_epochs=n_epochs, early_stopping=False)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(range(1,n_epochs+1), train_losses)
    ax.set_title(f'ResNet (n=2, {n_epochs} epochs)')
    fig.savefig('q1_part2.pdf')

def q1_part3(cifar10_path):
    splits = {
        'train': [f'{cifar10_path}/data_batch_{i}' for i in range(1,5)],
        'val': [f'{cifar10_path}/data_batch_5'],
        'test': [f'{cifar10_path}/test_batch']
        }
    datasets = {split: Cifar10Dataset(batch_files) for split, batch_files in splits.items()}

    n_epochs = 50
    if DEBUG:
        datasets = {split: Subset(dataset, np.arange(debug_len[split])) for split, dataset in datasets.items()}
        n_epochs = 5

    dataloaders = {split: DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True) for split, dataset in datasets.items()}
    model = ResNet(2, 10).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss_fn = nn.CrossEntropyLoss()

    best_model, (train_losses, val_losses) = train_model(model, optimizer, scheduler, dataloaders, loss_fn, max_epochs=n_epochs, early_stopping=True)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(range(1,len(train_losses)+1), train_losses, label='train')
    ax.plot(range(1,len(train_losses)+1), val_losses, label='val')
    ax.legend()
    ax.set_title(f'ResNet (n=2, {len(train_losses)} epochs, early stopping)')
    fig.savefig('q1_part3.pdf')

    eval_model(model, dataloaders)
    torch.save(best_model, 'part_1.1.pth')

    pickle.dump(train_losses, open('pt1_train_curve.pkl', 'wb'))
    pickle.dump(val_losses, open('pt1_val_curve.pkl', 'wb'))

def q2_part1(cifar10_path):
    norm_classes = {
        'bn': BatchNormalizer,
        'in': InstanceNormalizer,
        'bin': BatchInstanceNormalizer,
        'ln': LayerNormalizer,
        'gn': GroupNormalizer,
        'nn': nn.Identity
    }

    splits = {
        'train': [f'{cifar10_path}/data_batch_{i}' for i in range(1,5)],
        'val': [f'{cifar10_path}/data_batch_5'],
        'test': [f'{cifar10_path}/test_batch']
        }
    datasets = {split: Cifar10Dataset(batch_files) for split, batch_files in splits.items()}

    n_epochs = 50
    if DEBUG:
        datasets = {split: Subset(dataset, np.arange(debug_len[split])) for split, dataset in datasets.items()}
        n_epochs = 5

    dataloaders = {split: DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True) for split, dataset in datasets.items()}
    train_curves = {}
    val_curves = {}

    for norm_name, norm_class in norm_classes.items():
        model = ResNet(2, 10, batch_norm_class=norm_class).to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        loss_fn = nn.CrossEntropyLoss()

        best_model, (train_losses, val_losses) = train_model(
                model, optimizer, scheduler, dataloaders, loss_fn, 
                max_epochs=n_epochs, early_stopping=True, clip_rho=(norm_name == 'bin'))
        train_curves[norm_name] = train_losses
        val_curves[norm_name] = val_losses

        print(f'Model {norm_name}:')
        eval_model(model, dataloaders)
        torch.save(best_model, f'part_1.2_{norm_name}.pth')

    pickle.dump(train_curves, open('pt2_train_curves.pkl', 'wb'))
    pickle.dump(val_curves, open('pt2_val_curves.pkl', 'wb'))

if __name__ == "__main__":

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    #q1_part2(cifar10_path='data')
    q1_part3(cifar10_path='data')
    q2_part1(cifar10_path='data')
    

