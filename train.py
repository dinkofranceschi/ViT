import torch
import torch.nn as nn
from model import VisionTransformer, Transformer, Performer
import torch.optim as optim
import torchvision
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import datetime
import json
import os
from timm.loss import LabelSmoothingCrossEntropy

embed_dim=384
num_heads=6
dim_feedforward=1536
enc_layers=2
img_size=32
num_classes=10
patch_size=4
in_chans=1
smoothing=0.1
device='cuda:0'
lr=1e-4
epochs=100
clip_norm=1
batch_size=8
attention='performer'
kernel='relu'

def build_model():
    if attention == 'transformer':
        transformer=Transformer(
            d_model= embed_dim,
            dropout=0.1,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            normalize_before=True,
        )
    elif attention == 'performer':
        transformer = Performer(
            d_model= embed_dim,
            dropout=0.1,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            normalize_before=True,
        )

    model = VisionTransformer(img_size=img_size,
                              patch_size=patch_size,
                              in_chans=in_chans,
                              num_classes=num_classes,
                              num_queries=num_classes//10,
                              transformer=transformer,
                              shuffle=False)

    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing).to(device)
    
    optimizer= optim.AdamW(model.parameters(),lr=lr)

    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=1e-2) 
    
    return model,criterion, optimizer,scheduler


def build_dataset(dataset,batch_size):
    #Data
    if dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                    #torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1),interpolation=3),
                                    torchvision.transforms.Resize(32,interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
          batch_size=batch_size, shuffle=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                        torchvision.transforms.Resize(32,interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
          batch_size=batch_size, shuffle=True)
    elif dataset == 'CIFAR100':
        
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1),interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=batch_size, shuffle=True,pin_memory=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=batch_size, shuffle=True,pin_memory=True)
    
    elif dataset == 'CIFAR10':
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR10('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=batch_size, shuffle=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=batch_size, shuffle=True)
    
    elif dataset =='ImageNet':
          train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.ImageNet('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ])),
          batch_size=batch_size, shuffle=True)
        
          valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.ImageNet('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(32),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ])),
              batch_size=batch_size, shuffle=True)
          
    elif dataset == "CIFAR100_224":
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(224,scale=(0.7,1),interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=batch_size, shuffle=True,pin_memory=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(224,interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=batch_size, shuffle=True,pin_memory=True)
    
    return train_loader,valid_loader


def training(model,criterion,optimizer,scheduler,train_loader,valid_loader,epochs,clip_norm):
    log={'train_loss':[],
         'train_accuracy':[],
         'val_loss':[],
         'val_accuracy':[],
         'learning_rate':[]}
    max_norm= clip_norm
    model.train()
    #print(model)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
    
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            #print(output.shape,label.shape)
            loss = criterion(output, label)
    
            optimizer.zero_grad()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            #REMOVE
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
        scheduler.step()
        new_lr=scheduler.get_last_lr()[0] 
        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)
                val_output = model(data)
                val_loss = criterion(val_output, label)
    
                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n -curr_lr: {scheduler.get_last_lr()[0]}"
        )
        log['train_loss']=log['train_loss']+[epoch_loss.item()]
        log['train_accuracy']=log['train_accuracy']+[epoch_accuracy.item()]
        log['val_loss']=log['val_loss']+[epoch_val_loss.item()]
        log['val_accuracy']=log['val_accuracy']+[epoch_val_accuracy.item()]
        log['learning_rate']=log['learning_rate']+scheduler.get_last_lr()

    
    
        
def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

def main():
    model,criterion,optimizer,scheduler = build_model()
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"n_parameters={n_parameters}")
    criterion.to(device)
    train_loader,valid_loader=build_dataset('MNIST',batch_size)
    print('Start training')
    start_time=time.time()
    training(model,criterion,optimizer,scheduler,train_loader,valid_loader,epochs,clip_norm)    
    total_time=time.time()-start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    
main()
    
    